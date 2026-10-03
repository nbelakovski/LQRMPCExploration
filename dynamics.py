from constants import (
    r_arm_m,
    m_arm_kg,
    I_arm_kgm2,
    efficiency,
    bearing_friction_torque_Nm,
    viscous_friction_Nms,
    g_mps2,
    GEAR_RATIO
)
import numpy as np

'''
The dynamics are derived from the equation T=I * w_dot (https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics))
where

T = Torque applied to the system by both external forces and our control
I = Moment of inertia of the arm (modeled as a point mass, so mr^2)
and
w_dot = omega_dot is the angular acceleration

The system is modeled with ϴ being the angle between the arm and the horizontal position,
like so:

  /  
 /  ϴ  
/______

The torque applied to the system by external forces includes gravity and friction.

The gravity term is simply r X F (r cross F) where F=mg, so r X F = rmg cos(theta).
Typically a cross product uses the sin of the angle between the vectors, but in this
case theta is 90-(the angle between r and F), hence the usage of cos.

There's three types of friction:
1) Inefficiencies in gears and chains. We'll assume an efficiency of 98% for each stage.
2) Sliding friction from bearings. This should be low since that's what bearings are for.
3) Viscous friction from rolling bearings. This should also be fairly low.

For the moment of inertia, we can model the arm as a point mass which gives m*r^2 for
the moment of interia, and the rotor inside the motor has an inertia as well. We can
estimate that from the dimension and weight of the motor.


'''

# Moments of intertia (MoI)

# I'm not including the MoIs of the various gears and sprockets and the chain. This is
# an area where the model could be improved.

# Estimating rotor inertia from physical parameters
# Source: https://docs.wcproducts.com/welcome/electronics/kraken-x60/kraken-x60-motor/overview-and-features/physical-specifications
radius_motor_m = 0.06 - .005  # Assume motor wall is about 5mm thick
m_motor_kg = 0.54 * 0.9  # Assume the rotor is 90% of the total mass
I_motor_kgm2 = 1/2 * m_motor_kg * radius_motor_m**2  # Assume solid cylinder about z axis
# Lastly, the above needs to be reflected to the shaft with the arm so that we can write
# things in terms of the angle and rotational speed of the arm shaft. Reflected in this
# case means multiplying by the final gear ratio squared.

# Let's also add some model error in here, so that when we test the controllers we're
# testing against an imperfect model
model_error = 1.1  # 10%

def nonlinear_dynamics(t, x, u_Nm=0, friction=True):
    xdot = np.zeros(x.shape)
    efficiency_ = efficiency if friction else 0
    angle_rad, angular_rate_radps = x[0], x[1]
    dynamic_friction = bearing_friction_torque_Nm * np.tanh(1e3*angular_rate_radps) if friction else 0
    viscous_friction = viscous_friction_Nms * angular_rate_radps if friction else 0
    w_dot = (I_arm_kgm2 * model_error + I_motor_kgm2 * model_error * efficiency_ * GEAR_RATIO**2)**-1 * (
        u_Nm * efficiency_ * GEAR_RATIO  # Torque from motor
        - r_arm_m * m_arm_kg * model_error * g_mps2 * np.cos(angle_rad)  # Torque from gravity
        - dynamic_friction - viscous_friction
    )

    xdot[0] = angular_rate_radps
    xdot[1] = w_dot
    return xdot

# Linearization
# Both LQR and MPC require a linearization of the dynamics. A is derived by differentiating
# our nonlinear dynamics wrt the state, and B is derived by differentiating those same
# dynamics wrt the control.
I_eff_inv = (I_arm_kgm2 + I_motor_kgm2 * efficiency * GEAR_RATIO**2)**-1
# A is a 2x2 matrix and I'm trying to use spaces to make it clear which code corresponds
# to which element of A.
A = lambda theta: np.array([
    [0                                                      , 1],
    [I_eff_inv * r_arm_m * m_arm_kg * g_mps2 * np.sin(theta), -I_eff_inv  * viscous_friction_Nms]])
B = np.array([[0], [I_eff_inv * efficiency * GEAR_RATIO]])

if __name__ == "__main__":
    # As a quick sanity check of the dynamics, when the script is run by itself it will
    # simulate the dynamics without control and with and without friction.
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0, (tf:=50), 10000)
    friction = solve_ivp(nonlinear_dynamics, [0, tf], [np.radians(-45), 0],
                         t_eval=t_eval, args=(0, True), atol=1e-8, rtol=1e-8)
    frictionless = solve_ivp(nonlinear_dynamics, [0, tf], [np.radians(-45), 0],
                             t_eval=t_eval, args=(0, False), atol=1e-8, rtol=1e-8)
    plt.plot(t_eval, np.degrees(friction.y[0, :]), 'b', label='Friction')
    plt.plot(t_eval, np.degrees(frictionless.y[0, :]), 'g', label='Frictionless')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.show()