import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from controllers import CONTROLLERS
from estimators import ESTIMATORS
from systems import SYSTEMS
from util.outs import make_out_dir


def draw(fig, system, xs, xhats, us):
    fig.clf()
    ax = fig.add_subplot(1, 2, 1, projection=system.projection)
    system.render(ax, xs)
    t = system.dt * np.arange(len(xs))
    ax = fig.add_subplot(2, 2, 2)
    for i in range(system.n):
        (line,) = ax.plot(t, xs[:, i], label=f"x{i}")
        ax.plot(t, xhats[:, i], "--", color=line.get_color())
    ax.set_ylabel("state (--: estimate)")
    ax.legend(loc="upper right", fontsize="small")
    ax = fig.add_subplot(2, 2, 4)
    if len(us):
        ax.plot(t[:-1], us)
    ax.set_ylabel("control")
    ax.set_xlabel("time [s]")
    fig.tight_layout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=SYSTEMS, default="double_integrator")
    p.add_argument("--controller", choices=CONTROLLERS, default="lqr")
    p.add_argument("--estimator", choices=ESTIMATORS, default="perfect")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--render_every", type=int, default=10, help="0 disables rendering")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    system = SYSTEMS[args.system]()
    controller = CONTROLLERS[args.controller](system)
    steps = args.steps or system.steps

    x = system.x0.copy()
    estimator = None
    if ESTIMATORS[args.estimator] is not None:
        # initial guess from the first measurement, unmeasured states set to zero
        xhat0 = np.linalg.pinv(system.C) @ system.measure(x, rng)
        estimator = ESTIMATORS[args.estimator](system, xhat0)

    if args.render_every:
        plt.ion()
        fig = plt.figure(figsize=(12, 6))

    xs, xhats, ys, us = [x], [], [], []
    for k in range(steps):
        xhat = x if estimator is None else estimator.xhat
        u = controller(xhat, k)
        y = system.measure(x, rng)
        if estimator is not None:
            estimator.step(u, y)
        x = system.dynamics(x, u) + rng.multivariate_normal(np.zeros(system.n), system.W)
        xhats.append(xhat)
        ys.append(y)
        us.append(u)
        xs.append(x)
        if args.render_every and k % args.render_every == 0:
            draw(fig, system, np.array(xs), np.array(xhats + [xhat]), np.array(us))
            plt.pause(0.001)
    xhats.append(x if estimator is None else estimator.xhat)

    xs, xhats, ys, us = map(np.array, (xs, xhats, ys, us))
    settings = {**vars(args), "steps": steps, "dt": system.dt}
    out = make_out_dir(f"{args.system}_{args.controller}_{args.estimator}_seed{args.seed}")
    np.savez(out / "traj.npz", xs=xs, xhats=xhats, ys=ys, us=us, dt=system.dt)
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    if not args.render_every:
        fig = plt.figure(figsize=(12, 6))
    draw(fig, system, xs, xhats, us)
    fig.savefig(out / "plot.png")
    print(f"final error: {np.linalg.norm(system.error(xs[-1], system.reference(steps)[0])):.3e}, saved {out}")

    if args.render_every:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
