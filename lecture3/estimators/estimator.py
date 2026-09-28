from abc import ABC, abstractmethod


class Estimator(ABC):
    def __init__(self, system, xhat0):
        self.system = system
        self.xhat = xhat0

    @abstractmethod
    def step(self, u, y):
        """Update and return xhat_{k+1} given control u_k and measurement y_k."""
