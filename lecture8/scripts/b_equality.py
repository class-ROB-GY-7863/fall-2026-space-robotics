"""Part B: equality constraint, primal-dual with and without dual regularization.
uv run python -m scripts.b_equality
"""

import matplotlib.pyplot as plt
import numpy as np

from problems import EqualityQP
from solvers import error_matrix, primal_dual_equality
from util.outs import make_out_dir


def main():
    p = EqualityQP()
    s_max = np.linalg.svd(p.D, compute_uv=False).max()
    x_star, _ = p.fixed_point(0.0)
    print(f"m = {p.m:.3f}, M = {p.M:.3f}, sigma_max(D) = {s_max:.3f}")

    # (eta, alpha): plain primal-dual, then the lecture choice alpha = mu / ell^2
    runs = [(0.0, 0.05)]
    for eta in (1.0, 0.1):
        mu, ell = min(p.m, eta), max(p.M, eta) + s_max
        runs.append((eta, mu / ell**2))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    iters = 5000
    for eta, alpha in runs:
        A = error_matrix(p.Q, p.D, alpha, eta)
        rho = max(abs(np.linalg.eigvals(A)))
        norm_A = np.linalg.norm(A, 2)
        x_eta, nu_eta = p.fixed_point(eta)
        zs = primal_dual_equality(p.grad_f, p.h, p.jac_h, np.zeros(p.n), np.zeros(p.p), alpha, eta, iters)
        err = np.linalg.norm(zs - np.concatenate([x_eta, nu_eta]), axis=1)
        label = f"eta = {eta:g}, alpha = {alpha:.4f}"
        (line,) = axs[0].semilogy(err, label=label)
        msg = f"{label}: rho(A) = {rho:.4f}, ||A|| = {norm_A:.4f}"
        if eta > 0:
            mu, ell = min(p.m, eta), max(p.M, eta) + s_max
            c = np.sqrt(1 - 2 * alpha * mu + alpha**2 * ell**2)
            axs[0].semilogy(err[0] * c ** np.arange(iters + 1), "--", color=line.get_color(), lw=1)
            msg += f", bound c = {c:.5f}"
        print(msg + f", ||x_eta - x*|| = {np.linalg.norm(x_eta - x_star):.3f}, ||h(x_eta)|| = {np.linalg.norm(p.h(x_eta)):.3f}")
        axs[1].semilogy(np.linalg.norm(zs[:, : p.n] - x_star, axis=1), color=line.get_color(), label=label)
    axs[0].set_title("error to own fixed point  (--: bound c^k)")
    axs[0].set_xlabel("iteration k")
    axs[0].legend()
    axs[1].set_title("distance to the true solution x*")
    axs[1].set_xlabel("iteration k")
    axs[1].legend()
    fig.tight_layout()

    out = make_out_dir("b_equality")
    fig.savefig(out / "plot.png")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
