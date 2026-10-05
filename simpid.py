'''
This is an exploration of using numerical integration to estimate preliminary PID values
for the pivoting arm on FRC team 516's 2026 competition bot.

Run a simulation with:  uv run main.py pid
'''

import numpy as np
from constants import KRAKEN_X60_Kt_FOC_Nmpa, KRAKEN_X60_Kt_Nmpa


def make_controller(x_ref, Kp=300, Kd=75, Kg=12, foc=True):
    '''
    Build a PIDF controller that drives the arm to x_ref.
    '''
    def controller(x):
        angle_rad, angular_rate_radps = x
        control_effort_A = (Kg*np.cos(angle_rad)
                            - Kp*(angle_rad - x_ref[0])
                            - Kd*angular_rate_radps)
        return control_effort_A * KRAKEN_X60_Kt_FOC_Nmpa if foc else KRAKEN_X60_Kt_Nmpa

    return controller
