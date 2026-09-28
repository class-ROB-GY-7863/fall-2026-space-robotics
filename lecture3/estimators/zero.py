from estimators.estimator import Estimator


class Zero(Estimator):
    """Open loop (dead reckoning): L = 0, xhat_{k+1} = f(xhat_k, u_k), ignores y."""

    def step(self, u, y):
        self.xhat = self.system.dynamics(self.xhat, u)
        return self.xhat
