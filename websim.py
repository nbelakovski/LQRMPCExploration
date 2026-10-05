'''
Entry point for the browser build. The page calls run() through Pyodide.

Everything of substance is imported from the same modules main.py uses, so the code
running in the browser is the code running at the command line. All this file adds is
turning numpy arrays into something JavaScript can plot.
'''

import numpy as np

import main
from constants import KRAKEN_X60_MAX_TORQUE_FOC_Nm


def run(controller, x_ref_deg=main.TARGET_ANGLE_DEG, x0_deg=main.INITIAL_ANGLE_DEG,
        tf_s=1.5, dt_control_ms=None, friction=True, **gains):
    '''Simulate one run and return plain lists, ready for the page to draw.'''
    spec = main.CONTROLLERS[controller]
    dt_control_s = dt_control_ms / 1000.0 if dt_control_ms else spec['dt_control_s']
    x_ref = np.array([np.radians(x_ref_deg), 0.0])

    controller_fn = main.build_controller(controller, x_ref, gains)
    t_s, angle_rad, rate_radps, torque_cmd_Nm = main.simulate(
        controller_fn, [np.radians(x0_deg), 0.0], tf_s, dt_control_s, friction)

    result = {
        't': t_s.tolist(),
        'angle_deg': np.degrees(angle_rad).tolist(),
        'rate_degps': np.degrees(rate_radps).tolist(),
        'torque_Nm': torque_cmd_Nm.tolist(),
        'x_ref_deg': None if controller == 'none' else x_ref_deg,
        'torque_limit_Nm': KRAKEN_X60_MAX_TORQUE_FOC_Nm,
    }
    # LQR hands its design back alongside the controller, and the page shows it.
    if hasattr(controller_fn, 'K'):
        result['K'] = controller_fn.K.ravel().tolist()
        result['eigenvalues'] = sorted(controller_fn.eigenvalues.real.tolist())
    return result
