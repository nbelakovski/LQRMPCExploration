'''
This is an exploration of how to do LQR based controller design for the pivoting arm on FRC team
516's competition bot.

Note that with LQR we can't put constraints on u, so nothing stops the designed gain from
asking for more torque than the Kraken can deliver. Since the design of MPC involves
solving a constrained optimization problem, that approach ought to be more applicable in
this case.

Run a simulation with:  uv run main.py lqr
'''

import numpy as np
from scipy.linalg import solve_continuous_are
from constants import (
    m_arm_kg,
    r_arm_m,
    efficiency,
    g_mps2,
    GEAR_RATIO,
)
from dynamics import A, B


def make_controller(x_ref, Q11=100, Q22=1, R=0.1, control_point_rad=np.pi/2):
    '''
    Build an LQR controller, plus a gravity feedforward term, that drives the arm to x_ref.

    Q11 prices angle error, Q22 prices rate error, and R prices control effort. The
    returned controller carries the designed gain and the closed-loop eigenvalues as
    attributes so that callers may examine them.
    '''
    A_cp = A(control_point_rad)

    # Check the controllability matrix and assert that the linearized system is controllable.
    # For an n-state system that matrix is [B, AB, A^2 B, ... A^(n-1) B], so with two states
    # it is just [B, AB].
    controllability = np.hstack([B, A_cp @ B])
    assert np.linalg.matrix_rank(controllability) == 2

    Q = np.array([[Q11, 0], [0, Q22]])
    # R is 1x1 because the arm has a single input, the motor torque. Keeping it a matrix
    # rather than a scalar lets the gain below be written the way the textbooks write it.
    R = np.array([[R]])

    # LQR finds the gain K that minimizes the infinite-horizon quadratic cost
    #
    #   J = integral from 0 to inf of (x' Q x + u' R u) dt
    #
    # Q prices how much we dislike being away from the setpoint and R prices how much we
    # dislike spending control effort, so their ratio is the only thing that really matters.
    # The minimizer comes out of the algebraic Riccati equation (ARE)
    #
    #   A' S + S A - S B R^-1 B' S + Q = 0
    #
    # which scipy solves for S. The optimal gain is then K = R^-1 B' S. Substituting
    # u = -Kx into xdot = Ax + Bu leaves the closed loop running as xdot = (A - BK)x, so the
    # eigenvalues of A - BK say how quickly the controller hauls the state back to the
    # setpoint. Both are negative here, which is what makes the closed loop stable.
    S = solve_continuous_are(A_cp, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    E = np.linalg.eigvals(A_cp - B @ K)

    # The controller design incorporates a gravity feedforward component and then the
    # K matrix gains from LQR.
    def controller(x):
        feedforward_Nm = m_arm_kg*g_mps2*r_arm_m*np.cos(x[0]) / (GEAR_RATIO*efficiency)
        return (feedforward_Nm - K @ (x - x_ref))[0]

    controller.K = K
    controller.eigenvalues = E
    controller.holding_torque_Nm = m_arm_kg*g_mps2*r_arm_m
    return controller
