'''
This is an exploration of using numerical integration to estimate preliminary PID values
for the pivoting arm on FRC team 516's competition bot.

My conclusion is that the usefulness is a little bit limited by the fact that we're not
including any damping term. In real life we have a P of 300 and a D of 75 and that works pretty
well, and lower values make it hard to close steady state error, but here a P/D of 160/20 works,
alongside a lower kG. It could be useful as a teaching tool and to get some values to start with.
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from constants import m_kg, r_m, g_mps2, GEAR_RATIO, KRAKEN_X60_MAX_TORQUE_Nm

def current_A_to_torque_Nm(current_A):
    # Convert current to speed, and speed to torque, based off the performance curves
    free_speed_rpm = 5800
    stall_current_A = 483
    free_current_A = 2
    stall_torque_Nm = 9.37
    current_speed_slope = (free_current_A-stall_current_A)/free_speed_rpm
    current_speed_intercept = stall_current_A
    # current_A = current_speed_slope * speed + current_speed_intercept
    speed = (current_A - current_speed_intercept)/current_speed_slope
    torque_speed_slope = -stall_torque_Nm/free_speed_rpm
    torque_speed_intercept = stall_torque_Nm
    torque_Nm = torque_speed_slope * speed + torque_speed_intercept
    return torque_Nm


x_ref = [np.pi/4, 0]
Kp = 160
Kd = 20
Kg = 7.86
def u(x):
    angle, angular_rate = x
    control_effort_A = Kg * m_kg*g_mps2*r_m*np.cos(angle) - Kp*(angle - x_ref[0]) - Kd*angular_rate
    # But this comes out in amps for torque current foc control, so I need to map amps back to torque?
    control_effort_Nm = current_A_to_torque_Nm(control_effort_A)
    return np.clip(control_effort_Nm, -KRAKEN_X60_MAX_TORQUE_Nm, KRAKEN_X60_MAX_TORQUE_Nm)
# The above u calculates torque required at the pivot, but we have a 60:1 gear ratio, so we need to divide the resultant u by 60
# to obtain the torque with which we will need to control our motor.
def f_nonlinear_control(t, x):
    return [x[1], GEAR_RATIO*u(x) -g_mps2/r_m * np.cos(x[0])]

sol = solve_ivp(f_nonlinear_control, [0, tf:=2], [0, 0], t_eval=np.linspace(0, tf, 1000), max_step = 0.001, rtol=1, atol=1)

fig, axs = plt.subplots(2, 1, layout="constrained")
fig.suptitle('Single robotic arm controlled by PID')
axs[0].plot(sol.t, np.degrees(sol.y[0, :]))
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Angle (deg)')
axs[0].hlines(np.degrees(x_ref[0]), 0, sol.t[-1], color='k', linestyle='--', label='Target angle')
axs[0].grid();
axs[0].legend();
axs[1].plot(sol.t, [u(sol.y[:, i]) for i in range(len(sol.t))])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control effort (Nm)')
axs[1].hlines(-KRAKEN_X60_MAX_TORQUE_Nm, 0, sol.t[-1], color='k', linestyle='--', label='Kraken X60 max torque')
axs[1].hlines(KRAKEN_X60_MAX_TORQUE_Nm, 0, sol.t[-1], color='k', linestyle='--')
axs[1].grid()
axs[1].legend()
plt.show()
