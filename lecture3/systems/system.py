from abc import ABC, abstractmethod

import numpy as np


class System(ABC):
    """Discrete-time system x_{k+1} = f(x_k, u_k) + w_k, y_k = C x_k + v_k.

    Subclasses set in __init__:
        n, m        state and control dimensions
        dt, steps   timestep and default simulation length
        x0, x_ref   initial and goal state
        Q, R        LQR state and control costs
        C, W, V     measurement matrix, process and sensor noise covariances
    """

    projection = None  # matplotlib projection used by render, e.g. "3d"

    @abstractmethod
    def dynamics(self, x, u):
        """Return x_{k+1} = f(x_k, u_k)."""

    @abstractmethod
    def render(self, ax, xs):
        """Draw the state history xs (k+1, n) onto ax."""

    def linearize(self, x_bar, u_bar, eps=1e-6):
        """Return A, B with x_{k+1} ~ f(x_bar, u_bar) + A (x - x_bar) + B (u - u_bar).

        Default: central finite differences. Override with analytic Jacobians.
        """
        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.m))
        for i in range(self.n):
            d = np.zeros(self.n)
            d[i] = eps
            A[:, i] = (self.dynamics(x_bar + d, u_bar) - self.dynamics(x_bar - d, u_bar)) / (2 * eps)
        for i in range(self.m):
            d = np.zeros(self.m)
            d[i] = eps
            B[:, i] = (self.dynamics(x_bar, u_bar + d) - self.dynamics(x_bar, u_bar - d)) / (2 * eps)
        return A, B

    def measure(self, x, rng=None):
        """Return y = C x + v, v ~ N(0, V). Noiseless if rng is None."""
        y = self.C @ x
        if rng is not None:
            y = y + rng.multivariate_normal(np.zeros(len(y)), self.V)
        return y

    def reference(self, k):
        """Return (x_ref, u_ref) at step k. Default: regulate to a fixed goal."""
        return self.x_ref, np.zeros(self.m)

    def error(self, x, x_ref):
        return x - x_ref
