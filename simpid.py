'''
This is an exploration of using numerical integration to estimate preliminary PID values
for the pivoting arm on FRC team 516's 2026 competition bot.

Run a simulation with:  uv run main.py pid
'''

import numpy as np


def current_A_to_torque_Nm(current_A, foc=True):
    # Convert current to torque, based off the performance curves at
    # https://docs.wcproducts.com/welcome/electronics/kraken-x60/kraken-x60-motor/overview-and-features/motor-performance
    free_speed_torque_Nm = 0  # Torque at free speed is 0
    if foc:
        free_speed_current_A = 2
        stall_torque_Nm = 9.37
        stall_current_A = 483
    else:
        free_speed_current_A = 2
        stall_torque_Nm = 7.09
        stall_current_A = 366

    slope = (stall_torque_Nm - free_speed_torque_Nm) / (stall_current_A - free_speed_current_A)
    intercept = stall_torque_Nm - slope * stall_current_A
    torque_Nm = slope * current_A + intercept
    return torque_Nm


def make_controller(x_ref, Kp=300, Kd=75, Kg=12, foc=True):
    '''
    Build a PIDF controller that drives the arm to x_ref.
    '''
    def controller(x):
        angle_rad, angular_rate_radps = x
        control_effort_A = (Kg*np.cos(angle_rad)
                            - Kp*(angle_rad - x_ref[0])
                            - Kd*angular_rate_radps)
        return current_A_to_torque_Nm(control_effort_A, foc)

    return controller
