'''
This is an exploration of how to do LQR based controller design for the pivoting arm on FRC team
516's competition bot.
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct
from constants import m_kg, r_m, g_mps2, I_kgm2, GEAR_RATIO


# The nonlinear dynamics:
f_nonlinear_no_control = lambda t, x: [x[1], -g_mps2/r_m * np.cos(x[0])]
# sol = solve_ivp(f_nonlinear_no_control, [0, tf:=30], [-np.pi/4, 0], t_eval=np.linspace(0, tf, 1000))
# plt.plot(sol.t, np.degrees(sol.y[0, :]))
# plt.grid();
# plt.show();
#
#

# General formula for linearization. Provide theta as a point to linearize about.
A = lambda theta: np.array([[0, 1], [g_mps2/r_m * np.sin(theta), 0]])
B = np.array([[0], [GEAR_RATIO/I_kgm2]])

# Check the controllability matrix and assert that the linearized system is controllable
C = ct.ctrb(A(np.pi/2), B)
assert np.linalg.matrix_rank(C) == 2

cp = np.pi/2  # control point


Q = np.array([[100, 0], [0, 1]])
R = 0.1

K, S, E = ct.lqr(A(cp), B, Q, R)

print("K:", K)
print("Eigenvalues:", E)
print("mgr (Nm):", m_kg*g_mps2*r_m)

x_ref = [np.pi/4, 0]

f_linear = lambda t, x: A(cp)@x + B@(m_kg*g_mps2*r_m*np.cos(x[0]) - K@(x - x_ref))
u = lambda x: m_kg*g_mps2*r_m*np.cos(x[0]) - K@(x - x_ref)
# The above u calculates torque required at the pivot, but we have a 60:1 gear ratio, so we need to divide the resultant u by 60
# to obtain the torque with which we will need to control our motor.
f_nonlinear_control = lambda t, x: [x[1], GEAR_RATIO*u(x)[0] -g_mps2/r_m * np.cos(x[0])]

sol = solve_ivp(f_nonlinear_control, [0, tf:=1], [0, 0], t_eval=np.linspace(0, tf, 1000))

fig, axs = plt.subplots(2, 1, layout="constrained")
fig.suptitle('Single robotic arm controlled by LQR')
axs[0].plot(sol.t, np.degrees(sol.y[0, :]))
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Angle (deg)');
axs[0].hlines(np.degrees(x_ref[0]), 0, sol.t[-1], color='k', linestyle='--', label='Target angle')
axs[0].grid();
axs[0].legend();
axs[1].plot(sol.t, [u(sol.y[:, i])[0] for i in range(len(sol.t))])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control effort (Nm)');
axs[1].hlines(7, 0, sol.t[-1], color='k', linestyle='--', label='Kraken X60 max torque')
axs[1].grid();
axs[1].legend();
plt.show();

# TODO: We need to put constraints on u. MPC seems the right approach for this
