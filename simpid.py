'''
This is an exploration of using numerical integration to estimate preliminary PID values
for the pivoting arm on FRC team 516's competition bot.

My conclusion is that the usefulness is a little bit limited by the fact that we're not
including any damping term. In real life we have a P of 300 and a D of 75 and that works pretty
well, and lower values make it hard to close steady state error, but here a P/D of 160/20 works,
alongside a lower kG. It could be useful as a teaching tool and to get some values to start with.
'''

import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from constants import KRAKEN_X60_MAX_TORQUE_FOC_Nm
from dynamics import nonlinear_dynamics

def current_A_to_torque_Nm(current_A, foc=True):
    # Convert current to torque, based off the performance curves at
    # https://docs.wcproducts.com/welcome/electronics/kraken-x60/kraken-x60-motor/overview-and-features/motor-performance
    free_torque_Nm = 0  # Torque at free speed is 0
    if foc:
        free_current_A = 2
        stall_torque_Nm = 9.37
        stall_current_A = 483
    else:
        free_current_A = 2
        stall_torque_Nm = 7.09
        stall_current_A = 366

    slope = (stall_torque_Nm - free_torque_Nm) / (stall_current_A - free_current_A)
    intercept = stall_torque_Nm - slope * stall_current_A
    torque_Nm = slope * current_A + intercept
    return torque_Nm


x_ref = [np.pi/4, 0]
# In this season's code, the Kp, Kd, and Kg values are 300, 75, and 12
#
Kp = 300
Kd = 10
Kg = 6.924
def pidf(x):
    angle_rad, angular_rate_radps = x
    control_effort_A = Kg*np.cos(angle_rad) - Kp*(angle_rad - x_ref[0]) - Kd*angular_rate_radps
    control_effort_Nm = current_A_to_torque_Nm(control_effort_A)
    # We need to clip the resultant control effort from the motor, bcause even if we request infinite amps, it can
    # only deliver so much torque
    control_effort_Nm = np.clip(control_effort_Nm, -KRAKEN_X60_MAX_TORQUE_FOC_Nm, KRAKEN_X60_MAX_TORQUE_FOC_Nm)
    return control_effort_Nm

f_nonlinear_dynamics_pidf_control = lambda t, x: nonlinear_dynamics(t, x, pidf(x))

initial_angle_rad = 0
initial_angular_rate_radps = 0
sol = RK45(f_nonlinear_dynamics_pidf_control,
    t0=0,
    y0=[initial_angle_rad, initial_angular_rate_radps],
    t_bound=(tf:=2),
    max_step = 0.001, rtol=1, atol=1
)

time = []
arm_angle_rad = []
motor_torque_Nm = []
while sol.status == 'running':
    sol.step()
    time.append(sol.t)
    arm_angle_rad.append(sol.y[0])
    motor_torque_Nm.append(pidf(sol.y))


fig, axs = plt.subplots(2, 1, layout="constrained")
fig.suptitle('Single robotic arm controlled by PID')
axs[0].plot(time, np.degrees(arm_angle_rad))
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Angle (deg)')
axs[0].hlines(np.degrees(x_ref[0]), 0, tf, color='k', linestyle='--', label='Target angle')
axs[0].grid()
axs[0].legend()
axs[1].plot(time, motor_torque_Nm)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Motor torque (Nm)')
axs[1].hlines(-KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, tf, color='k', linestyle='--', label='Kraken X60 max torque')
axs[1].hlines(KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, tf, color='k', linestyle='--')
axs[1].grid()
axs[1].legend()
plt.show()
