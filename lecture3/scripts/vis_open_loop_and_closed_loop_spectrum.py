"""Random (A, B): reject non-stabilizable, compute K via DARE, plot eig(A) and eig(A - BK).

    uv run python -m scripts.vis_open_loop_and_closed_loop_spectrum
"""

import matplotlib.pyplot as plt
import numpy as np

from util.dare import dare, is_stabilizable
from util.outs import make_out_dir


def random_pairs(num, rng):
    """Sparse random (A, B) so that some pairs are not stabilizable."""
    pairs, rejected = [], 0
    while len(pairs) < num:
        n, m = rng.integers(2, 7), rng.integers(1, 3)
        A = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.4)
        B = rng.normal(size=(n, m)) * (rng.random((n, m)) < 0.4)
        if is_stabilizable(A, B):
            pairs.append((A, B))
        else:
            rejected += 1
    return pairs, rejected


def main():
    pairs, rejected = random_pairs(50, np.random.default_rng(0))
    print(f"accepted {len(pairs)}, rejected {rejected} non-stabilizable pairs")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, layout="constrained")
    for i, (A, B) in enumerate(pairs):
        _, K = dare(A, B, np.eye(len(A)), np.eye(B.shape[1]))
        for ax, M in zip(axs, (A, A - B @ K)):
            lam = np.linalg.eigvals(M)
            ax.plot(lam.real, lam.imag, ".", color=f"C{i % 10}", alpha=0.7)
        assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1

    th = np.linspace(0, 2 * np.pi, 200)
    for ax, title in zip(axs, ("open loop: eig(A)", "closed loop: eig(A - BK)")):
        ax.plot(np.cos(th), np.sin(th), "k-")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("Re")
    axs[0].set_ylabel("Im")

    out = make_out_dir("spectrum") / "spectrum.png"
    fig.savefig(out)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
