from controllers.controller import Controller
from util.dare import dare


class LQR(Controller):
    """u = u_ref - K (x - x_ref), with K from DARE on the linearization about the reference."""

    def __call__(self, x, k):
        s = self.system
        x_ref, u_ref = s.reference(k)
        A, B = s.linearize(x_ref, u_ref)
        _, K = dare(A, B, s.Q, s.R)
        return u_ref - K @ s.error(x, x_ref)
