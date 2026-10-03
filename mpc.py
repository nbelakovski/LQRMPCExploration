'''
This is an exploration of how to do MPC based controller design for the pivoting arm on FRC team
516's competition bot.

Math is based on example 2.5 from Model Predictive Control: Theory, Computation, and Design
by Rawlings, Maybe, and Diehl
'''

import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.linalg import block_diag
from scipy.signal import cont2discrete
from constants import (
    m_arm_kg,
    r_arm_m,
    I_arm_kgm2,
    viscous_friction_Nms,
    efficiency,
    g_mps2,
    GEAR_RATIO,
    KRAKEN_X60_MAX_TORQUE_FOC_Nm
)
from dynamics import nonlinear_dynamics, I_motor_kgm2

# The nonlinear dynamics:
f_nonlinear_control = lambda t, x, u, params: nonlinear_dynamics(t, x, u[0])

# General formula for linearization. Provide theta as a point to linearize about.
I_eff_inv = (I_arm_kgm2 + I_motor_kgm2 * efficiency * GEAR_RATIO**2)**-1
# General formula for linearization. Provide theta as a point to linearize about.
A = lambda theta: np.array([[0, 1], [I_eff_inv * r_arm_m * m_arm_kg * g_mps2 * np.sin(theta), -I_eff_inv  * viscous_friction_Nms]])
B = np.array([[0], [I_eff_inv * efficiency * GEAR_RATIO]])


Q = np.array([[100, 0], [0, 1]])
R = 0.001
x_ref = [np.pi/4, 0]
dynamics = lambda t, x: np.append(f_nonlinear_control(t, x[:2], [x[2]], ()), 0)  # Need to append 0 since we include control as a state variable and we need its derivative to be 0
# Set rtol/atol to something high so that it respects the max step and basically acts like a fixed step RK45
iterator = RK45(dynamics, t0=0, y0=[0, 0, 0], t_bound=1.5, max_step=0.001, rtol=1, atol=1)

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

        Ac = A(iterator.y[0])
        Bc = np.array([[0], [I_eff_inv * efficiency * GEAR_RATIO]])
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
            # I believe that canonical way of doing this is to recalculate F at the new
            # state at the new time horizon. Its affect on accuracy in this sim will be
            # negligible, but I should do it for the sake of doing it right.
            constraint_mtrx.append([left_zeros, F, B_discrete, -np.eye(2), right_zeros])
        constraint_mtrx = np.block(constraint_mtrx)
        xref_vector = [*x_ref, 0] * (time_horizon_i - 1) + [*x_ref]
        b = -constraint_mtrx @ xref_vector - np.tile(c_d, time_horizon_i - 1)
        # Need to add the initial conditions to the constraint
        constraint_mtrx = np.vstack((constraint_mtrx, np.block([np.eye(2), np.zeros((2, time_horizon_i*3-1-2))])))
        b = np.concat((b, (iterator.y[:2] - x_ref)))  # Remember the state variables are actually x - xref, not x
        # Create the initial guess
        guess = np.array([*(iterator.y[:2] - x_ref), -0.1] * (time_horizon_i - 1) + [*(iterator.y[:2] - x_ref)])
        # Create the lb/ub
        lb = [-np.inf, -np.inf, -KRAKEN_X60_MAX_TORQUE_FOC_Nm] * (time_horizon_i - 1) + [-np.inf, -np.inf]
        lb = np.array(lb)
        ub = [np.inf, np.inf, KRAKEN_X60_MAX_TORQUE_FOC_Nm] * (time_horizon_i - 1) + [np.inf, np.inf]
        ub = np.array(ub)
        # Let'r rip
        x = solve_qp(P=np.array(P),
                     q=np.zeros((3*time_horizon_i - 1, 1)),
                     G=None,
                     h=None,
                     A=constraint_mtrx,
                     b=b,
                     lb=lb,
                     ub=ub,
                     solver='qpalm',
                     initvals=guess,
                     verbose=True
        )
        if x is None:
            raise Exception("No solution found")
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
axs[0].set_ylabel('Angle (deg)')
axs[0].hlines(np.degrees(x_ref[0]), 0, time_steps[-1], color='k', linestyle='--', label='Target angle')
axs[0].grid()
axs[0].legend()
axs[1].plot(time_steps, control)
axs[1].set_title("Kraken motor control effort required (gear ratio 60)")
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control (Nm)')
axs[1].hlines(KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, time_steps[-1], color='k', linestyle='--', label='Kraken X60 max torque FOC')
axs[1].hlines(-KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, time_steps[-1], color='k', linestyle='--')
axs[1].grid()

plt.show()
