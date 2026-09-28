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
from constants import m_kg, r_m, g_mps2, GEAR_RATIO, KRAKEN_X60_MAX_TORQUE_FOC_Nm



# The nonlinear dynamics:
f_nonlinear_control = lambda t, x, u, params: [x[1], -g_mps2*np.cos(x[0])/r_m - (u[0]*GEAR_RATIO)/(m_kg*r_m**2)]

nl_sys = ct.nlsys(f_nonlinear_control, inputs=1, states=2)
print(nl_sys)

constr = LinearConstraint(A=np.array([[0, 0, 1]]), lb=-KRAKEN_X60_MAX_TORQUE_FOC_Nm, ub=KRAKEN_X60_MAX_TORQUE_FOC_Nm)

Q = np.array([[100, 0], [0, 1]])
R = 0.001
x_ref = [np.pi/4, 0]
cost = lambda x, u: (x - x_ref)@Q@(x - x_ref) + R * u[0]**2
terminal_cost = lambda x, u: (x[0] - x_ref[0])**2
result = opt.solve_optimal_trajectory(nl_sys, np.linspace(0, 0.5, 5), [0, 0], cost, constr, terminal_cost=terminal_cost)
print(result)

# Set rtol/atol to something high so that it respects the max step and basically acts like a fixed step RK45
iterator = RK45(lambda t, x: np.append(f_nonlinear_control(t, x[:2], [x[2]], ()), 0), t0:=0, [0, 0, 0], tf:=1.5, max_step=0.001, rtol=1, atol=1)

angle = []
control = []
time_steps = []
last_solve_s = 0
while iterator.status == 'running':
    msg = iterator.step()
    if iterator.t - last_solve_s > 0.01:
        # solve and update u
        result = opt.solve_optimal_trajectory(nl_sys, np.linspace(0, 0.5, 5), iterator.y[:2], cost, constr, terminal_cost=terminal_cost, initial_guess=iterator.y[2], print_summary=False)
        iterator.y[2] = result.inputs[0][0]
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
