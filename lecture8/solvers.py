"""Lecture 8 solvers: the algorithms from the lecture slides.

Fill in every block marked TODO. Do not change the function signatures.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Part A: unconstrained minimization
# ---------------------------------------------------------------------------

def gradient_descent(grad, x0, alpha, iters):
    """x_{k+1} = x_k - alpha * grad(x_k). Return all iterates."""
    xs = [np.array(x0, dtype=float)]
    for _ in range(iters):
        raise NotImplementedError("TODO: append the next iterate")
    return np.array(xs)


# ---------------------------------------------------------------------------
# Part B: equality constraints  min f(x)  s.t.  h(x) = 0
# ---------------------------------------------------------------------------

def primal_dual_equality(grad_f, h, jac_h, x0, nu0, alpha, eta, iters):
    """Primal-dual with dual regularization eta >= 0 (eta = 0: plain primal-dual).

        x_{k+1}  = x_k  - alpha * (grad_f(x_k) + jac_h(x_k)^T nu_k)
        nu_{k+1} = nu_k + alpha * (h(x_k) - eta * nu_k)

    jac_h(x) is the Jacobian (p x n). Return all augmented states z_k = (x_k, nu_k).
    """
    x, nu = np.array(x0, dtype=float), np.array(nu0, dtype=float)
    zs = [np.concatenate([x, nu])]
    for _ in range(iters):
        # TODO: the two update lines from the docstring (use the OLD x in both lines)
        raise NotImplementedError
        zs.append(np.concatenate([x, nu]))
    return np.array(zs)


def error_matrix(H, D, alpha, eta):
    """Error matrix A = I - alpha * G of primal_dual_equality, for f with Hessian H and h(x) = Dx - g:

        G = [[H, D^T], [-D, eta I]]
    """
    raise NotImplementedError("TODO: build G with np.block, return I - alpha * G")


# ---------------------------------------------------------------------------
# Part C: equality and inequality constraints  min f(x)  s.t.  h(x) = 0,  g(x) <= 0
# ---------------------------------------------------------------------------

def primal_dual_inequality(grad_f, h, jac_h, g, jac_g, x0, alpha, eta, iters):
    """Primal-dual with dual regularization eta > 0, for equality and inequality constraints.

        x_{k+1}   = x_k   - alpha * (grad_f(x_k) + jac_h(x_k)^T nu_k + jac_g(x_k)^T lam_k)
        nu_{k+1}  = nu_k  + alpha * (h(x_k) - eta * nu_k)
        lam_{k+1} = max(0, lam_k + alpha * (g(x_k) - eta * lam_k))     (max for each entry)

    Start from nu = 0, lam = 0. Return all iterates x_k.
    """
    x = np.array(x0, dtype=float)
    nu = np.zeros(len(h(x)))
    lam = np.zeros(len(g(x)))
    xs = [x.copy()]
    for _ in range(iters):
        # TODO: the three update lines from the docstring (use the OLD x in all lines)
        raise NotImplementedError
        xs.append(x.copy())
    return np.array(xs)
