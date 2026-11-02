"""Part A: gradient descent and its contraction rate.  uv run python -m scripts.a_unconstrained"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from problems import QuadLogistic
from solvers import gradient_descent
from util.outs import make_out_dir


def main():
    p = QuadLogistic()
    x0 = np.array([10.0, -10.0])
    x_star = minimize(p.f, x0, jac=p.grad, hess=p.hess, method="trust-exact", options={"gtol": 1e-14}).x
    kappa = p.M / p.m
    print(f"m = {p.m:.2f}, M = {p.M:.2f}, kappa = {kappa:.1f}")

    runs = {
        "alpha = 1/M": (1 / p.M, 1 - p.m / p.M),
        "alpha = 2/(m+M)": (2 / (p.m + p.M), (kappa - 1) / (kappa + 1)),
        "alpha = 2.1/M (too large)": (2.1 / p.M, None),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (alpha, rho) in runs.items():
        xs = gradient_descent(p.grad, x0, alpha, 300)
        err = np.linalg.norm(xs - x_star, axis=1)
        (line,) = ax.semilogy(np.clip(err, 1e-16, 1e8), label=name)
        if rho is not None:
            k = np.arange(len(err))
            ax.semilogy(k, err[0] * rho**k, "--", color=line.get_color(), lw=1)
            obs = (err[100] / err[50]) ** (1 / 50)
            print(f"{name}: predicted c = {rho:.4f}, observed rate = {obs:.4f}")
        else:
            print(f"{name}: final error = {err[-1]:.2e}")
    ax.set_xlabel("iteration k")
    ax.set_ylabel("||x_k - x*||  (--: bound c^k)")
    ax.set_title("Gradient descent on QuadLogistic")
    ax.legend()
    fig.tight_layout()

    out = make_out_dir("a_unconstrained")
    fig.savefig(out / "plot.png")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
