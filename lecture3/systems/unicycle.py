import numpy as np

from systems.system import System


class Unicycle(System):
    """Kinematic unicycle, x = (px, py, theta), u = (v, omega). Tracks a circle."""

    def __init__(self, v_ref=1.0, radius=3.0):
        self.n, self.m = 3, 2
        self.dt, self.steps = 0.05, 600
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.Q = np.diag([1.0, 1.0, 0.1])
        self.R = 0.1 * np.eye(2)
        self.C = np.eye(3)
        self.W = 1e-6 * np.eye(3)
        self.V = 1e-3 * np.eye(3)

        # reference: roll out the model itself so x_ref[k+1] = f(x_ref[k], u_ref)
        self.u_ref = np.array([v_ref, v_ref / radius])
        self.x_refs = [np.array([radius, 0.0, np.pi / 2])]
        for _ in range(self.steps):
            self.x_refs.append(self.dynamics(self.x_refs[-1], self.u_ref))
        self.x_refs = np.array(self.x_refs)
        self.x_ref = self.x_refs[0]

    def dynamics(self, x, u):
        px, py, th = x
        v, w = u
        return np.array([px + self.dt * v * np.cos(th), py + self.dt * v * np.sin(th), th + self.dt * w])

    def linearize(self, x_bar, u_bar):
        th, v = x_bar[2], u_bar[0]
        A = np.array([
            [1, 0, -self.dt * v * np.sin(th)],
            [0, 1, self.dt * v * np.cos(th)],
            [0, 0, 1],
        ])
        B = np.array([
            [self.dt * np.cos(th), 0],
            [self.dt * np.sin(th), 0],
            [0, self.dt],
        ])
        return A, B

    def reference(self, k):
        return self.x_refs[min(k, self.steps)], self.u_ref

    def error(self, x, x_ref):
        e = x - x_ref
        e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
        return e

    def render(self, ax, xs):
        ax.plot(self.x_refs[:, 0], self.x_refs[:, 1], "g--", alpha=0.5)
        ax.plot(xs[:, 0], xs[:, 1], "b-", alpha=0.5)
        k = len(xs) - 1
        x_ref, _ = self.reference(k)
        ax.plot(x_ref[0], x_ref[1], "g*", ms=15)
        px, py, th = xs[-1]
        ax.quiver(px, py, np.cos(th), np.sin(th), color="b", scale=15)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("px")
        ax.set_ylabel("py")


class UnicycleParking(Unicycle):
    """Park at the origin: v_bar = 0, so the linearization is not stabilizable."""

    def __init__(self):
        super().__init__()
        self.x0 = np.array([2.0, 1.0, 0.0])
        self.x_ref = np.zeros(3)
        self.u_ref = np.zeros(2)
        self.x_refs = np.tile(self.x_ref, (self.steps + 1, 1))
