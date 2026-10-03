'''
This is an exploration of how to do LQR based controller design for the pivoting arm on FRC team
516's competition bot.
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct
from constants import (
    m_arm_kg,
    r_arm_m,
    efficiency,
    g_mps2,
    GEAR_RATIO,
    KRAKEN_X60_MAX_TORQUE_FOC_Nm
)
from dynamics import nonlinear_dynamics, A, B


cp = np.pi/2  # control point

# Check the controllability matrix and assert that the linearized system is controllable
C = ct.ctrb(A(cp), B)
assert np.linalg.matrix_rank(C) == 2

Q = np.array([[100, 0], [0, 1]])
R = 0.1

K, S, E = ct.lqr(A(cp), B, Q, R)

print("K:", K)
print("Eigenvalues:", E)
print("mgr (Nm):", m_arm_kg*g_mps2*r_arm_m)

x_ref = [np.pi/4, 0]

# The controller design incorporates a gravity feedforward component and then the
# K matrix gains from LQR
u = lambda x: m_arm_kg*g_mps2*r_arm_m*np.cos(x[0])/(GEAR_RATIO*efficiency) - K@(x - x_ref)

nonlinear_dynamics_lqrf_control = lambda t, x: nonlinear_dynamics(t, x, u(x)[0])

sol = solve_ivp(nonlinear_dynamics_lqrf_control, [0, tf:=1], [0, 0], t_eval=np.linspace(0, tf, 1000))

fig, axs = plt.subplots(2, 1, layout="constrained")
fig.suptitle('Single robotic arm controlled by LQR')
axs[0].plot(sol.t, np.degrees(sol.y[0, :]))
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Angle (deg)')
axs[0].hlines(np.degrees(x_ref[0]), 0, sol.t[-1], color='k', linestyle='--', label='Target angle')
axs[0].grid()
axs[0].legend()
axs[1].plot(sol.t, [u(sol.y[:, i])[0] for i in range(len(sol.t))])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control effort (Nm)')
axs[1].hlines(KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, sol.t[-1], color='k', linestyle='--', label='Kraken X60 max torque FOC')
axs[1].hlines(-KRAKEN_X60_MAX_TORQUE_FOC_Nm, 0, sol.t[-1], color='k', linestyle='--')
axs[1].grid()
axs[1].legend()
plt.show()

# Note that with LQR, We can't put constraints on u.
# Since the design of MPC involves solving a constrained optimization problem,
# that approach ought to be more applicable in this case.
