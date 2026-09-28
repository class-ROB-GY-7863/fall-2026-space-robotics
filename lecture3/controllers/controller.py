from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def __call__(self, x, k):
        """Return control u_k given state (estimate) x at step k."""
