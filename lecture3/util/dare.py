import numpy as np
from scipy.linalg import solve_discrete_are


def is_stabilizable(A, B, tol=1e-9):
    """PBH test: rank [A - lam I, B] = n for every eigenvalue |lam| >= 1."""
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1 - tol and np.linalg.matrix_rank(np.hstack([A - lam * np.eye(n), B]), tol=1e-6) < n:
            return False
    return True


def dare(A, B, Q, R):
    """Solve P = Q + A'PA - A'PB (R + B'PB)^-1 B'PA, return P and K = (R + B'PB)^-1 B'PA."""
    if not is_stabilizable(A, B):
        raise ValueError("(A, B) is not stabilizable: DARE has no stabilizing solution")
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K
