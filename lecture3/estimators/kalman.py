from util.dare import dare
from estimators.estimator import Estimator


class Kalman(Estimator):
    """Steady-state (extended) Kalman filter in predictor form:
    xhat_{k+1} = f(xhat_k, u_k) + L (y_k - C xhat_k), with L from the dual DARE.
    """

    def step(self, u, y):
        s = self.system
        A, _ = s.linearize(self.xhat, u)
        _, K = dare(A.T, s.C.T, s.W, s.V)
        self.xhat = s.dynamics(self.xhat, u) + K.T @ (y - s.C @ self.xhat)
        return self.xhat
