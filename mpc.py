'''
This is an exploration of how to do MPC based controller design for the pivoting arm on FRC team
516's competition bot.
'''

import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import control as ct
import control.optimal as opt
from scipy.optimize import LinearConstraint
from qpsolvers import solve_qp
from scipy.linalg import block_diag
from scipy.signal import cont2discrete
from constants import m_kg, r_m, g_mps2, I_kgm2, GEAR_RATIO, KRAKEN_X60_MAX_TORQUE_Nm


# The nonlinear dynamics:
f_nonlinear_no_control = lambda t, x: [x[1], -g_mps2/r_m * np.cos(x[0])]
f_nonlinear_control = lambda t, x, u, params: [x[1], -g_mps2*np.cos(x[0])/r_m - (u[0]*GEAR_RATIO)/(m_kg*r_m**2)]

# General formula for linearization. Provide theta as a point to linearize about.
A = lambda theta: np.array([[0, 1], [g_mps2/r_m * np.sin(theta), 0]])
B = np.array([[0], [-GEAR_RATIO/I_kgm2]])

nl_sys = ct.nlsys(f_nonlinear_control, inputs=1, states=2)
print(nl_sys)


constr = LinearConstraint(A=np.array([[0, 0, 1]]), lb=-KRAKEN_X60_MAX_TORQUE_Nm, ub=KRAKEN_X60_MAX_TORQUE_Nm)

Q = np.array([[100, 0], [0, 1]])
R = 0.001
x_ref = [np.pi/4, 0]
cost = lambda x, u: (x - x_ref)@Q@(x - x_ref) + R * u[0]**2
def cost_slsqp(x):
    # TODO: This needs to be a sum of the costs at each stage.
    # x is a vector that looks like x11, x12, u1, x21, x22, u2, ... xn1, xn2
    # And we have the option of either adding the dynamics as constraints, or incorporating them
    # into the cost function by utilizing the fact that as difference equations we can just substitue
    # x1{1,2} with the difference equation depending on x0{1,2} and so forth. Let's try the option with
    # constraints first, since that feels a bit more natural in terms of how to present the problem
    # Will SLSQP require the Jacobian of the cost function?
    return (x[:2] - x_ref)@Q@(x[:2] - x_ref) + R * x[2]**2
terminal_cost = lambda x, u: (x[0] - x_ref[0])**2
result = opt.solve_optimal_trajectory(nl_sys, np.linspace(0, 0.5, 5), [0, 0], cost, constr, terminal_cost=terminal_cost)
print(result)

# Set rtol/atol to something high so that it respects the max step and basically acts like a fixed step RK45
iterator = RK45(lambda t, x: np.append(f_nonlinear_control(t, x[:2], [x[2]], ()), 0), t0:=0, [0, 0, 0], tf:=1.5, max_step=0.001, rtol=1, atol=1)

angle = []
control = []
time_steps = []
last_solve_s = 0
dt_s = 0.02
time_horizon_i = 10
B_discrete =np.array([[dt_s**2/2], [B[1,0]*dt_s]])
while iterator.status == 'running':
    msg = iterator.step()
    if iterator.t - last_solve_s > 0.01:
        ## solve and update u with python- control library
        # result = opt.solve_optimal_trajectory(nl_sys, np.linspace(0, 0.5, 5), iterator.y[:2], cost, constr, terminal_cost=terminal_cost, initial_guess=iterator.y[2], print_summary=False)
        # iterator.y[2] = result.inputs[0][0]
        ## New task: Replace opt.solve_optimal_trajectory with a QP solvable by a QP solver
        ## But why would I linearize and use a QP solver, if I can keep it nonlinear and use SLSQP? I genuinely don't know
        ## I mean, I don't know about the compute capabilities of the rio or, more importantly, the pi, i.e. if we can run SLSQP on it
        ## But maybe I start there, and if we can't get the necessary loop times, fall back to QP?
        ## And then there's the question of getting SLSQP in Java? There's an old slsqp4j package, might work?
        ## 7/25/26 I'm going to try to make it a QP problem just to get practice with QP
        # 1. Linearize the matrices
        # 2. Convert them to discrete time with the appropriate dt
        F = A(iterator.y[0])*dt_s + np.eye(2)
        F, B_discrete, _, _, _ = cont2discrete((A(iterator.y[0]), B, None, None), dt=0.1)

        Ac = A(iterator.y[0])
        Bc = np.array([[0.], [-GEAR_RATIO/I_kgm2]])
        x_bar, u_bar = iterator.y[:2], iterator.y[2]
        c = f_nonlinear_control(0, x_bar, [u_bar], ()) - Ac @ x_bar - Bc@[u_bar]

        F, Md, *_ = cont2discrete((Ac, np.hstack([Bc, c.reshape(2,1)]), None, None), dt=dt_s)
        B_discrete, c_d = Md[:, :1], Md[:, 1]

        # Create the cost matrix
        diag = [Q, R] * (time_horizon_i - 1) + [Q]
        P = block_diag(*diag)
        # Create the constraint matrix and b vector from constraint matrix and xref
        constraint_mtrx = []
        for i in range(time_horizon_i - 1):
            left_zeros = np.zeros((2, 3*i))
            right_zeros = np.zeros((2, time_horizon_i*3-1 - 5 - 3*i))  # F, B, -I takes up 5 cols
            constraint_mtrx.append([left_zeros, F, B_discrete, -np.eye(2), right_zeros])
        constraint_mtrx = np.block(constraint_mtrx)
        xref_vector = [*x_ref, 0] * (time_horizon_i - 1) + [*x_ref]
        b = -constraint_mtrx @ xref_vector - np.tile(c_d, time_horizon_i - 1)
        # Need to add the initial conditions to the constraint
        constraint_mtrx = np.vstack((constraint_mtrx, np.block([np.eye(2), np.zeros((2, time_horizon_i*3-1-2))])))
        b = np.concat((b, (iterator.y[:2] - x_ref)))  # Remember the state variables are actually x - xref, not x
        # Create the initial guess
        guess = [*(iterator.y[:2] - x_ref), -0.1] * (time_horizon_i - 1) + [*(iterator.y[:2] - x_ref)]
        # Create the lb/ub
        lb = [-np.inf, -np.inf, -KRAKEN_X60_MAX_TORQUE_Nm] * (time_horizon_i - 1) + [-np.inf, -np.inf]
        lb = np.array(lb)
        ub = [np.inf, np.inf, KRAKEN_X60_MAX_TORQUE_Nm] * (time_horizon_i - 1) + [np.inf, np.inf]
        ub = np.array(ub)
        # Let'r rip
        print("SOLVING")
        x = solve_qp(P=P,
                     q=np.zeros((3*time_horizon_i - 1, 1)),
                     G=None,
                     h=None,
                     A=constraint_mtrx,
                     b = b,
                     lb = lb,
                     ub = ub,
                     # ['clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'jaxopt_osqp', 'osqp', 'piqp', 'proxqp', 'qpalm', 'qpax', 'qtqp', 'quadprog', 'pyqpmad', 'scs', 'sip']
                     solver = 'qpalm',
                     initvals = guess,
                     verbose = True
        )
        print("Control: ", x[2])
        iterator.y[2] = x[2]
        last_solve_s = iterator.t
    angle.append(iterator.y[0])
    control.append(iterator.y[2])
    time_steps.append(iterator.t)

angle = np.array(angle)
control = np.array(control)
time_steps = np.array(time_steps)

fig, axs = plt.subplots(2, 1, layout="constrained")
fig.suptitle("Single robotic arm controlled by MPC")
axs[0].plot(time_steps, np.degrees(angle))
axs[0].set_title("Actuator angle over time. 0 is horizontal")
axs[0].set_ylabel('Angle (deg)');
axs[0].hlines(np.degrees(x_ref[0]), 0, time_steps[-1], color='k', linestyle='--', label='Target angle')
axs[0].grid();
axs[0].legend();
# plt.show();
axs[1].plot(time_steps, control)
axs[1].set_title("Kraken motor control effort required (gear ratio 60)")
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control (Nm)');
axs[1].grid();

plt.show();
