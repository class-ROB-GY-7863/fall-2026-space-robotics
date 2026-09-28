import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_discrete_are


# implement system from lecture 1

# x = [x,y,z,phi,theta,psi,vx,vy,vz,p,q,r]
# u = [fx,fy,fz,taux,tauy,tauz]


# =====================================================================
# Controller design: linearize -> discretize (Euler) -> LQR, exactly
# following the derivation from lecture (Clohessy-Wiltshire for
# translation, hover-linearization for attitude).
# =====================================================================

# Actuator limits (finite thrust/torque authority -- this is what makes
# "too large an initial error" actually fail to converge: the linear
# feedback law would stabilize ANY error given unlimited control
# authority, so the divergence we want to demonstrate is a combination
# of (a) actuator saturation and (b) the linearization becoming invalid
# far from the point it was taken about.)
MAX_THRUST = 100.0   # N per axis  (small-sat RCS cluster, e.g. ~1600 kg spacecraft)
MAX_TORQUE = 10.0    # N*m per axis


def get_cw_matrices(n):
    """Continuous-time Clohessy-Wiltshire A,B (Hill frame; radial x,
    along-track y, cross-track z), control input = force (N)."""
    return NotImplementedError


def get_attitude_matrices():
    """Continuous-time linearized attitude dynamics about hover
    (Theta_ref = 0, omega = 0): the gyroscopic cross term
    omega x (I omega) vanishes exactly to first order at omega=0."""
    return NotImplementedError


def discretize_euler(Ac, Bc, dt):
    """Forward-Euler discretization: x_{k+1} = (I + Ac*dt) x_k + (Bc*dt) u_k."""
    return NotImplementedError


def dlqr(A, B, Q, R):
    """Discrete-time LQR gain via the DARE (same P also certifies
    Lyapunov stability of the closed loop, as derived in lecture)."""
    return NotImplementedError


def design_translation_controller(param, dt):
    return NotImplementedError


def design_attitude_controller(param, dt):
    return NotImplementedError


def make_controller(param, K_trans, K_att, n, a):
    """Combine translation (Hill-frame CW) + attitude (hover) feedback
    into the full u = [f_ext; tau_ext] expected by dxdt."""

    return NotImplementedError


# =====================================================================
# Simulation
# =====================================================================

def simulate(fn, dv_err=np.zeros(3), dr_err=np.zeros(3),
             theta0_deg=(0.0, 0.0, 0.0), omega0=np.zeros(3),
             T_min=90.0, dt=1.0, use_controller=True):

    param = {
        "mass": 1600,          # kg
        "radius": 2.4,         # m
        "height": 6.2,         # m
        "mass_earth": 5.97219e24,   # kg
        "r_earth": np.array([0, 0, 0]),
        "radius_earth": 6371e3,     # m
        "G": 6.67430e-11,
    }
    Ix = 1.0 / 12.0 * param["mass"] * (3 * param["radius"] ** 2.0 + param["height"] ** 2.0)
    Iz = 1.0 / 2.0 * param["mass"] * param["radius"] ** 2.0
    param["inertia"] = np.array([[Ix, 0, 0], [0, Ix, 0], [0, 0, Iz]])

    # reference circular orbit radius = nominal initial position magnitude
    param["a_ref"] = param["radius_earth"] + 500e3

    # --- design controllers (linearize -> Euler discretize -> LQR) ---
    if use_controller:
        K_trans, n = design_translation_controller(param, dt)
        K_att = design_attitude_controller(param, dt)
        controller_fn = make_controller(param, K_trans, K_att, n, param["a_ref"])
    else:
        n = np.sqrt(param["G"] * param["mass_earth"] / param["a_ref"] ** 3)
        controller_fn = None

    # --- initial state: nominal circular orbit + requested error ---
    a = param["a_ref"]
    v_circ = a * n
    r0 = np.array([a, 0.0, 0.0]) + np.asarray(dr_err)
    v0 = np.array([0.0, v_circ, 0.0]) + np.asarray(dv_err)
    Theta0 = np.deg2rad(np.asarray(theta0_deg))
    x0 = np.concatenate([r0, Theta0, v0, np.asarray(omega0)])

    num_steps = int(T_min * 60 / dt)

    xs = [x0]
    ts = [0.0]
    us = []
    diverged_at = None
    for step in range(num_steps):
        t = step * dt
        x_cur = xs[-1]
        if controller_fn is not None:
            u, dr_H, dv_H = controller_fn(x_cur, t)
        else:
            u = np.zeros(6)
        us.append(u)
        x_next = x_cur + dxdt(x_cur, u, param) * dt

        if not np.all(np.isfinite(x_next)):
            diverged_at = t + dt
            print(f"!! state diverged (non-finite) at t={diverged_at:.1f} s -- stopping early")
            break

        xs.append(x_next)
        ts.append((step + 1) * dt)

    xs = np.array(xs)
    ts = np.array(ts)

    save(fn, xs, ts, param, diverged_at)
    return ts, xs, param, diverged_at


def dxdt(x, u, param):
    # x is np array in (12,)
    # u is np array in (6,)

    _dxdt = np.zeros((12,))

    # state
    r = x[0:3]
    Theta = x[3:6]
    v = x[6:9]
    omega = x[9:12]

    # control
    f_ext = u[0:3] / param["mass"]
    tau_ext = u[3:6]

    # gravity
    r_rel = param["r_earth"] - r
    f_g = param["G"] * param["mass_earth"] * r_rel / np.power(np.linalg.norm(r_rel), 3)

    # dynamics
    _dxdt[0:3] = v
    _dxdt[3:6] = B_inv(Theta) @ omega
    _dxdt[6:9] = f_g + f_ext
    _dxdt[9:12] = np.linalg.pinv(param["inertia"]) @ (- np.cross(omega, param["inertia"] @ omega) + tau_ext)

    return _dxdt


def B_inv(Theta):
    phi, theta, psi = Theta[0], Theta[1], Theta[2]
    return np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), - np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])


def save(fn, xs, ts, params, diverged_at):
    with h5py.File(fn, "w") as h5file:
        h5file.create_dataset("xs", data=xs)
        h5file.create_dataset("ts", data=ts)
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                h5file.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                h5file.attrs[key] = str(value)
        h5file.attrs["diverged_at"] = -1.0 if diverged_at is None else diverged_at


def load(fn):
    with h5py.File(fn, "r") as h5file:
        xs = h5file["xs"][:]
        ts = h5file["ts"][:]
        params = {key: h5file.attrs[key] for key in h5file.attrs}
    return ts, xs, params


# =====================================================================
# Plotting: translational error and attitude angles over time, for
# both a small and a large initial error, side by side.
# =====================================================================

def hill_frame_position_error(xs, ts, a, n):
    r = xs[:, 0:3]
    r_ref = a * np.stack([np.cos(n * ts), np.sin(n * ts), np.zeros_like(ts)], axis=1)
    return np.linalg.norm(r - r_ref, axis=1)


def plot_comparison(fn_small, fn_large, out_path="./translation_attitude_convergence.png"):
    ts_s, xs_s, params_s = load(fn_small)
    ts_l, xs_l, params_l = load(fn_large)

    a = params_s["a_ref"]
    mu = params_s["G"] * params_s["mass_earth"]
    n = np.sqrt(mu / a ** 3)

    dr_s = hill_frame_position_error(xs_s, ts_s, a, n)
    dr_l = hill_frame_position_error(xs_l, ts_l, a, n)

    theta_s_deg = np.rad2deg(np.linalg.norm(xs_s[:, 3:6], axis=1))
    theta_l_deg = np.rad2deg(np.linalg.norm(xs_l[:, 3:6], axis=1))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

    ax = axes[0]
    ax.plot(ts_s / 60.0, dr_s, label="small initial error (converges)", color="tab:green")
    ax.plot(ts_l / 60.0, dr_l, label="large initial error (diverges)", color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("||relative position error|| (m, log scale)")
    ax.set_title("Translational error vs. time (Hill-frame distance from reference orbit)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(ts_s / 60.0, theta_s_deg, label="small initial error (converges)", color="tab:green")
    ax.plot(ts_l / 60.0, theta_l_deg, label="large initial error (diverges)", color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("||attitude error|| (deg, log scale)")
    ax.set_title("Attitude error vs. time (deviation from hover)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    plt.show()


def render(fn):
    ts, xs, params = load(fn)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color="blue")
    stepsize = max(1, len(xs) // 20)
    for step in range(0, len(xs), stepsize):
        plot_body_frame(ax, xs[step, 0:3], xs[step, 3], xs[step, 4], xs[step, 5], scale=1e6)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = params["radius_earth"] * np.outer(np.cos(u), np.sin(v))
    y = params["radius_earth"] * np.outer(np.sin(u), np.sin(v))
    z = params["radius_earth"] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="green", alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Spacecraft Trajectory with Earth")
    plt.show()


def euler_to_rotation_matrix(phi, theta, psi):
    R_x = np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return R_z @ R_y @ R_x


def plot_body_frame(ax, position, phi, theta, psi, scale=1.0):
    orientation = euler_to_rotation_matrix(phi, theta, psi)
    x_body, y_body, z_body = orientation[:, 0], orientation[:, 1], orientation[:, 2]
    ax.quiver(position[0], position[1], position[2], x_body[0], x_body[1], x_body[2], color="red", length=scale)
    ax.quiver(position[0], position[1], position[2], y_body[0], y_body[1], y_body[2], color="green", length=scale)
    ax.quiver(position[0], position[1], position[2], z_body[0], z_body[1], z_body[2], color="blue", length=scale)


if __name__ == '__main__':
    # SMALL initial error: modest velocity/attitude offset -- the
    # linearization (CW for translation, hover for attitude) is valid
    # here, and the LQR-designed feedback drives the error to (near) zero.
    print("=== Running SMALL initial error case ===")
    simulate(
        "./spacecraft_small_error.h5",
        dv_err=np.array([0.0, 10.0, 2.0]),      # m/s off circular velocity
        dr_err=np.array([0.0, 0.0, 0.0]),
        theta0_deg=(5.0, 3.0, -2.0),
        omega0=np.array([0.001, 0.0, 0.0]),
        T_min=90.0, dt=1.0,
    )

    # LARGE initial error: the same linear feedback law is used, but the
    # error is now big enough that (a) actuator authority saturates and
    # (b) the CW/hover linearization is no longer a good local model --
    # the closed loop fails to converge and the state diverges.
    print("\n=== Running LARGE initial error case ===")
    simulate(
        "./spacecraft_large_error.h5",
        dv_err=np.array([0.0, 150.0, 20.0]),    # m/s off circular velocity
        dr_err=np.array([50000.0, 0.0, 0.0]),   # 50 km radial offset
        theta0_deg=(60.0, 45.0, -50.0),
        omega0=np.array([0.05, 0.0, 0.02]),
        T_min=90.0, dt=1.0,
    )

    plot_comparison("./spacecraft_small_error.h5", "./spacecraft_large_error.h5",
                     out_path="./translation_attitude_convergence.png")

    # Optional 3D trajectory render (small-error case)
    # render("./spacecraft_small_error.h5")