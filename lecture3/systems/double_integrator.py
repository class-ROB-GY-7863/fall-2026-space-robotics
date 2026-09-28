import numpy as np

from systems.system import System


class DoubleIntegrator(System):
    """2d double integrator, x = (px, py, vx, vy), u = (ax, ay)."""

    def __init__(self):
        self.n, self.m = 4, 2
        self.dt, self.steps = 0.1, 200
        self.x0 = np.array([5.0, 3.0, 0.0, -2.0])
        self.x_ref = np.zeros(4)
        self.Q = np.eye(4)
        self.R = np.eye(2)
        self.C = np.hstack([np.eye(2), np.zeros((2, 2))])  # measure position only
        self.W = 1e-5 * np.eye(4)
        self.V = 1e-2 * np.eye(2)

    def dynamics(self, x, u):
        A, B = self.linearize(x, u)
        return A @ x + B @ u

    def linearize(self, x_bar, u_bar):
        A = np.eye(4)
        A[0:2, 2:4] = self.dt * np.eye(2)
        B = np.vstack([np.zeros((2, 2)), self.dt * np.eye(2)])
        return A, B

    def render(self, ax, xs):
        ax.plot(xs[:, 0], xs[:, 1], "b-", alpha=0.5)
        ax.plot(self.x_ref[0], self.x_ref[1], "g*", ms=15)
        ax.plot(xs[-1, 0], xs[-1, 1], "bo", ms=10)
        ax.set_xlim(-2, 7)
        ax.set_ylim(-3, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("px")
        ax.set_ylabel("py")
