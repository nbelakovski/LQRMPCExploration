'''
Constants of the pivot arm for the 516 bot

Mostly physical parameters.
'''


from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry()

# There are 3 stages between the motor and the arm shaft, I'm assuming the gears and
# chain are 98% efficient at transferring the load.
eta_s1 = 0.98
eta_s2 = 0.98
eta_chain = 0.98
efficiency = eta_s1 * eta_s2 * eta_chain

# Bearing friction
# The bearings will have some friction. I count 6 bearings in the CAD, so I will just
# assume 0.01 for their total.
bearing_friction_torque_Nm = 0.01

# There may also be some viscous friction from the grease inside the bearings. I'll
# estimate this as 0.01 Nm when we're at 2pi rad/s. This could be refined.
viscous_friction_Nms = 0.01/(2*np.pi)

m_arm_kg = 3.023
r_arm_in = 7.640
r_arm_m = ureg.convert(r_arm_in, ureg.inch, ureg.m)
g_mps2 = 9.81
# I'm estimating the arm's MoI by taking its mass from CAD as well as the distance from
# its center of mass to the pivot point, and then I treat it like a point mass at that
# distance.
I_arm_kgm2 = m_arm_kg * r_arm_m**2
GEAR_RATIO = 60

KRAKEN_X60_MAX_TORQUE_FOC_Nm = 9.37
