'''
Run one of the controllers against the nonlinear arm model and plot the result.

This is the only file that knows about matplotlib, about the integration loop, and about
the setpoint. The controller modules (simpid, lqr, mpc_control) each just hand back a
callable that maps the arm's state to a motor torque, so all three can be driven by the
identical loop below and compared on equal footing.

    uv run main.py none --tf 20
    uv run main.py pid
    uv run main.py lqr --Q11 400 --R 0.01
    uv run main.py mpc --x-ref 60 --horizon-points 7
'''

import argparse

import numpy as np
from scipy.integrate import RK45

from constants import KRAKEN_X60_MAX_TORQUE_FOC_Nm
from dynamics import nonlinear_dynamics

# The integrator always takes 1ms steps, independently of how often the controller is
# allowed to speak. A real robot's control loop is slower than the physics it is
# controlling, and holding these apart is what lets us see that.
INTEGRATOR_MAX_STEP_S = 0.001

# Where the arm starts and where it is asked to go, both measured up from horizontal.
# Every run starts from rest at INITIAL_ANGLE_DEG.
INITIAL_ANGLE_DEG = 22.0
TARGET_ANGLE_DEG = 102.0

# Per-controller defaults: which gains it accepts and how often it runs. The gain
# *values* deliberately live with the controllers themselves, not here.
CONTROLLERS = {
    # No controller at all, for looking at what the arm does on its own. dt_control_s is
    # meaningless here, since the thing being called every tick always answers zero.
    'none': dict(gains=(), dt_control_s=0.010),
    'pid': dict(gains=('Kp', 'Kd', 'Kg'), dt_control_s=0.001),
    'lqr': dict(gains=('Q11', 'Q22', 'R'), dt_control_s=0.010),
    'mpc': dict(gains=('Q11', 'Q22', 'R', 'horizon_s', 'horizon_points'),
                dt_control_s=0.010),
}


def build_controller(name, x_ref, gains):
    '''Import the chosen controller module and ask it for a controller.

    The imports are done here rather than at the top of the file so that running the PID
    does not pay for importing python-control, which drags in matplotlib and takes a
    couple of seconds.
    '''
    if name == 'none':
        # Nothing to import and nothing to tune: the arm is left to gravity and friction.
        # Running it through the same loop as the other three is what makes the
        # no-control plot directly comparable to theirs.
        return lambda x: 0.0
    if name == 'pid':
        import simpid
        return simpid.make_controller(x_ref, **gains)
    if name == 'lqr':
        import lqr
        return lqr.make_controller(x_ref, **gains)
    if name == 'mpc':
        import mpc_control
        return mpc_control.make_controller(x_ref, **gains)
    raise ValueError(f'unknown controller {name!r}')


def apply_motor_limit(u_Nm):
    '''What the Kraken actually delivers when asked for u_Nm.'''
    return float(np.clip(u_Nm,
                         -KRAKEN_X60_MAX_TORQUE_FOC_Nm, KRAKEN_X60_MAX_TORQUE_FOC_Nm))


def simulate(controller, x0, tf_s, dt_control_s, friction=True):
    '''Integrate the arm forward, letting the controller update every dt_control_s.

    The commanded torque rides along as a third state with a derivative of zero, so it is
    held constant across every step the integrator takes and only changes where we assign
    to it below. That is what a motor controller does with a commanded value, and it means
    the integrator never sees the control change partway through one of its own steps.

    Every controller is allowed to ask for whatever torque it likes; the motor's limit is
    applied here, between the controller and the plant.
    '''
    u_cmd_Nm = controller(np.asarray(x0))

    integrator = RK45(
        lambda t, x: np.append(nonlinear_dynamics(t, x[:2], x[2], friction), 0),
        t0=0, y0=[x0[0], x0[1], apply_motor_limit(u_cmd_Nm)], t_bound=tf_s,
        # rtol/atol are deliberately loose so that max_step is what governs the step size
        # and this behaves like a fixed-step integrator.
        max_step=INTEGRATOR_MAX_STEP_S, rtol=1, atol=1,
    )

    t_s = [0.0]
    angle_rad = [x0[0]]
    rate_radps = [x0[1]]
    torque_cmd_Nm = [u_cmd_Nm]

    last_control_s = 0.0
    while integrator.status == 'running':
        integrator.step()
        # The epsilon keeps a dt_control_s that is an exact multiple of the step size from
        # being missed by a hair and firing a step late.
        if integrator.t - last_control_s >= dt_control_s - 1e-12:
            u_cmd_Nm = controller(integrator.y[:2])
            integrator.y[2] = apply_motor_limit(u_cmd_Nm)
            last_control_s = integrator.t
        t_s.append(integrator.t)
        angle_rad.append(integrator.y[0])
        rate_radps.append(integrator.y[1])
        torque_cmd_Nm.append(u_cmd_Nm)

    return (np.array(t_s), np.array(angle_rad),
            np.array(rate_radps), np.array(torque_cmd_Nm))


def plot(t_s, angle_rad, torque_cmd_Nm, x_ref, title, subtitle):
    '''Angle and control effort against time.

    x_ref is None for the no-control run, which has no target to draw.
    '''
    # Imported here rather than at the top of the file so that everything above can be
    # used without matplotlib present. The browser build leans on that: it reuses
    # simulate() and build_controller() and draws the results itself.
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, layout='constrained', sharex=True)
    fig.suptitle(f'{title}\n{subtitle}', fontsize=10)

    axs[0].plot(t_s, np.degrees(angle_rad))
    axs[0].set_ylabel('Angle (deg)')
    axs[0].grid()
    if x_ref is not None:
        axs[0].axhline(np.degrees(x_ref[0]), color='k', linestyle='--',
                       label='Target angle')
        axs[0].legend()

    axs[1].plot(t_s, torque_cmd_Nm)
    axs[1].axhline(KRAKEN_X60_MAX_TORQUE_FOC_Nm, color='k', linestyle='--',
                   label='Kraken X60 max torque FOC')
    axs[1].axhline(-KRAKEN_X60_MAX_TORQUE_FOC_Nm, color='k', linestyle='--')
    axs[1].set_ylabel('Control effort (Nm)')
    axs[1].set_xlabel('Time (s)')
    axs[1].grid()
    axs[1].legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('controller', choices=sorted(CONTROLLERS))
    parser.add_argument('--x-ref', type=float, default=TARGET_ANGLE_DEG, metavar='DEG',
                        help='target arm angle above horizontal, in degrees '
                             '(default: %(default)s)')
    parser.add_argument('--x0', type=float, default=INITIAL_ANGLE_DEG, metavar='DEG',
                        help='angle the arm starts from, at rest (default: %(default)s)')
    parser.add_argument('--tf', type=float, default=1.5, metavar='S',
                        help='simulation duration in seconds (default: %(default)s)')
    parser.add_argument('--no-friction', action='store_true',
                        help='switch off bearing and viscous friction and gear losses, '
                             'to see how much of the behaviour they account for')
    parser.add_argument('--dt-control', type=float, metavar='MS',
                        help='how often the controller runs, in milliseconds '
                             '(default: 1 for pid, 10 for lqr and mpc)')

    gains = parser.add_argument_group(
        'gains', 'Left unset, each controller uses its own defaults.')
    gains.add_argument('--Kp', type=float, help='PID proportional gain (amps/rad)')
    gains.add_argument('--Kd', type=float, help='PID derivative gain (amps/(rad/s))')
    gains.add_argument('--Kg', type=float, help='PID gravity feedforward (amps)')
    gains.add_argument('--Q11', type=float, help='LQR/MPC cost on angle error')
    gains.add_argument('--Q22', type=float, help='LQR/MPC cost on rate error')
    gains.add_argument('--R', type=float, help='LQR/MPC cost on control effort')
    gains.add_argument('--horizon-s', type=float, help='MPC lookahead, in seconds')
    gains.add_argument('--horizon-points', type=int, help='MPC points across the horizon')

    args = parser.parse_args()
    spec = CONTROLLERS[args.controller]

    # Only pass along the gains the user actually set, so the defaults that matter stay
    # written down in the controller modules.
    all_gain_names = ('Kp', 'Kd', 'Kg', 'Q11', 'Q22', 'R', 'horizon_s', 'horizon_points')
    supplied = {name: getattr(args, name)
                for name in all_gain_names if getattr(args, name) is not None}
    irrelevant = sorted(set(supplied) - set(spec['gains']))
    if irrelevant:
        takes = (', '.join('--' + g for g in spec['gains'])
                 if spec['gains'] else 'no gains at all')
        parser.error(f'{args.controller} does not take '
                     + ', '.join(f'--{n.replace("_", "-")}' for n in irrelevant)
                     + f' (it takes {takes})')

    # With no controller there is nothing aiming at a setpoint, so x_ref stops being
    # meaningful and the plot leaves the target line off.
    tracks_setpoint = args.controller != 'none'
    x_ref = np.array([np.radians(args.x_ref), 0.0])
    dt_control_s = (args.dt_control / 1000.0 if args.dt_control is not None
                    else spec['dt_control_s'])

    controller = build_controller(args.controller, x_ref, supplied)

    # LQR hands back its design alongside the controller; nothing else does.
    if hasattr(controller, 'K'):
        print('K:', controller.K)
        print('Closed-loop eigenvalues:', controller.eigenvalues)
        print('mgr (Nm):', controller.holding_torque_Nm)

    friction = not args.no_friction
    x0 = [np.radians(args.x0), 0.0]
    t_s, angle_rad, rate_radps, torque_cmd_Nm = simulate(
        controller, x0, args.tf, dt_control_s, friction)

    settled_deg = np.degrees(angle_rad[-1])
    if tracks_setpoint:
        print(f'settled at {settled_deg:.3f} deg '
              f'(error {settled_deg - args.x_ref:+.3f} deg), '
              f'peak |u| {np.abs(torque_cmd_Nm).max():.2f} Nm')
        shown_gains = ', '.join(f'{k}={v}' for k, v in supplied.items()) or 'default gains'
        title = f'Single robotic arm controlled by {args.controller.upper()}'
        subtitle = (f'{shown_gains}; target {args.x_ref:g} deg; '
                    f'control every {dt_control_s*1000:g} ms'
                    + ('' if friction else '; frictionless'))
    else:
        print(f'ended at {settled_deg:.3f} deg, '
              f'moving at {np.degrees(rate_radps[-1]):.1f} deg/s')
        title = 'Single robotic arm with no control input'
        subtitle = (f'released from rest at {args.x0:g} deg, '
                    + ('gravity and friction only' if friction else 'gravity only'))

    plot(t_s, angle_rad, torque_cmd_Nm,
         x_ref if tracks_setpoint else None, title=title, subtitle=subtitle)


if __name__ == '__main__':
    main()
