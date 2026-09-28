import numpy as np

from systems.system import System


class SingleIntegrator(System):
    """p_{k+1} = p_k + dt u_k"""

    def __init__(self):
        self.n, self.m = 1, 1
        self.dt, self.steps = 0.1, 100
        self.x0 = np.array([5.0])
        self.x_ref = np.array([0.0])
        self.Q = np.eye(1)
        self.R = np.eye(1)
        self.C = np.eye(1)
        self.W = 1e-4 * np.eye(1)
        self.V = 1e-2 * np.eye(1)

    def dynamics(self, x, u):
        return x + self.dt * u

    def linearize(self, x_bar, u_bar):
        return np.eye(1), self.dt * np.eye(1)

    def render(self, ax, xs):
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(self.x_ref[0], 0, "g*", ms=15)
        ax.plot(xs[-1, 0], 0, "bo", ms=10)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel("p")
