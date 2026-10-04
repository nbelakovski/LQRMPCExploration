'''
This is an exploration of how to do MPC based controller design for the pivoting arm on FRC team
516's competition bot, using python-control's trajectory optimization.

Unlike LQR, this one can be told about the motor's torque limit directly, as a constraint on
the optimization rather than as a clip applied after the fact.

Run a simulation with:  uv run main.py mpc

(mpc.py is the same idea written out as my own transcription instead of handed to
python-control.)
'''

import numpy as np
import control as ct
import control.optimal as opt
from scipy.optimize import LinearConstraint
from constants import KRAKEN_X60_MAX_TORQUE_FOC_Nm
from dynamics import nonlinear_dynamics


def make_controller(x_ref, Q11=100, Q22=1, R=0.001, horizon_s=0.5, horizon_points=5):
    '''
    Build an MPC controller that drives the arm to x_ref.

    Q11, Q22 and R price angle error, rate error and control effort exactly as they do for
    LQR, but here they weight a finite horizon that is re-optimized at every control tick
    rather than an infinite one solved once.

    Each call solves a fresh nonlinear trajectory optimization, which is far and away the
    most expensive thing in this project. horizon_s and horizon_points are the knobs that
    trade solve time against lookahead.
    '''
    # Refactor the nonlinear dynamics in a format consumable by python-control
    f_nonlinear_control = lambda t, x, u, params: nonlinear_dynamics(t, x, u[0])
    nl_sys = ct.nlsys(f_nonlinear_control, inputs=1, states=2)

    # Create the motor constraints. The [0, 0, 1] row picks u out of the stacked
    # (angle, rate, u) vector that the optimizer hands to the constraint.
    constr = LinearConstraint(A=np.array([[0, 0, 1]]),
                              lb=-KRAKEN_X60_MAX_TORQUE_FOC_Nm,
                              ub=KRAKEN_X60_MAX_TORQUE_FOC_Nm)

    Q = np.array([[Q11, 0], [0, Q22]])
    horizon = np.linspace(0, horizon_s, horizon_points)

    # Running and terminal cost functions
    cost = lambda x, u: (x - x_ref)@Q@(x - x_ref) + R * u[0]**2
    terminal_cost = lambda x, u: (x[0] - x_ref[0])**2

    # The previous solution, used to warm start the next solve. Holding it here means the
    # optimizer starts from somewhere sensible instead of from zero every tick.
    last_u_Nm = 0.0

    def controller(x):
        nonlocal last_u_Nm
        result = opt.solve_optimal_trajectory(
            nl_sys, horizon, x, cost, constr,
            terminal_cost=terminal_cost,
            initial_guess=last_u_Nm,
            print_summary=False,
        )
        # MPC plans the whole horizon but we only ever apply its first move, then re-solve
        # from wherever the arm actually ended up.
        last_u_Nm = result.inputs[0][0]
        return last_u_Nm

    return controller
