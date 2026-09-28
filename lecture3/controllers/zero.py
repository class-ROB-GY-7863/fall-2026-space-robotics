import numpy as np

from controllers.controller import Controller


class Zero(Controller):
    """Open loop: u = 0."""

    def __call__(self, x, k):
        return np.zeros(self.system.m)
