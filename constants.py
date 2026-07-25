'''
Constants of the pivot arm for the 516 bot

Mostly physical parameters.
'''


from pint import UnitRegistry

ureg = UnitRegistry()

m_kg = 3.023
r_in = 7.640
r_m = ureg.convert(r_in, ureg.inch, ureg.m)
g_mps2 = 9.81
I_kgm2 = m_kg * r_m**2
GEAR_RATIO = 60

KRAKEN_X60_MAX_TORQUE_Nm = 7
