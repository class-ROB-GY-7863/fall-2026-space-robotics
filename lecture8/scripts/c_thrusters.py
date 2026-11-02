"""Part C: energy-optimal thruster allocation with primal-dual (equality and inequality constraints).
uv run python -m scripts.c_thrusters

    min 1/2 ||u||^2   s.t.   B u = w,   0 <= u <= umax
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from problems import ThrusterQP
from solvers import primal_dual_inequality
from util.outs import make_out_dir


def main():
    p = ThrusterQP()
    # reference solution from a standard solver
    ref = minimize(lambda u: 0.5 * u @ u, np.zeros(p.n), jac=lambda u: u, method="SLSQP",
                   constraints=[{"type": "eq", "fun": p.h, "jac": p.jac_h}],
                   bounds=[(0, p.umax)] * p.n, options={"ftol": 1e-14, "maxiter": 500})
    u_star = ref.x
    u_pinv = np.linalg.pinv(p.B) @ p.w
    print(f"pseudo-inverse u = B^+ w has min entry {u_pinv.min():.3f} (negative thrust)")

    alpha, iters = 0.1, 5000
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for eta in (1.0, 0.1, 0.01):
        us = primal_dual_inequality(p.grad_f, p.h, p.jac_h, p.g, p.jac_g, np.zeros(p.n), alpha, eta, iters)
        u = us[-1]
        axs[0].semilogy(np.linalg.norm(us - u_star, axis=1), label=f"eta = {eta:g}")
        print(f"eta = {eta:5g}: ||u - u*|| = {np.linalg.norm(u - u_star):.4f}, "
              f"||Bu - w|| = {np.linalg.norm(p.h(u)):.4f}, min u = {u.min():.4f}")
    axs[0].set_xlabel("iteration k")
    axs[0].set_ylabel("||u_k - u*||")
    axs[0].legend()

    xs = np.arange(p.n)
    axs[1].bar(xs - 0.2, u, 0.4, label=f"primal-dual, eta = {eta:g}")
    axs[1].bar(xs + 0.2, u_star, 0.4, label="reference (SLSQP)")
    axs[1].set_xticks(xs, p.names, rotation=60)
    axs[1].set_ylabel("thrust [N]")
    axs[1].legend()
    fig.tight_layout()

    out = make_out_dir("c_thrusters")
    fig.savefig(out / "plot.png")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
