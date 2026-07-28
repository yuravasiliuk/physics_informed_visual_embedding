import numpy as np
from scipy.spatial.transform import Rotation


def _wxyz2xyzw(q):
    q = np.atleast_2d(q)
    return q[:, [1, 2, 3, 0]]


def _xyzw2wxyz(q):
    q = np.atleast_2d(q)
    return q[:, [3, 0, 1, 2]]


def mats_to_quats(R):
    return _xyzw2wxyz(Rotation.from_matrix(R).as_quat())


def cheb_basis(degree, tau):
    x = 2.0 * np.asarray(tau, float) - 1.0
    B = np.empty((len(x), degree + 1))
    B[:, 0] = 1.0
    if degree >= 1:
        B[:, 1] = x
    for k in range(2, degree + 1):
        B[:, k] = 2 * x * B[:, k - 1] - B[:, k - 2]
    return B


def fit_basis(tau, Y, degree):
    B = cheb_basis(degree, tau)
    coeffs, *_ = np.linalg.lstsq(B, Y, rcond=None)
    err = float(np.sqrt(np.mean((B @ coeffs - Y) ** 2)))
    return coeffs, err


def reconstruct(tau, coeffs):
    return cheb_basis(coeffs.shape[0] - 1, tau) @ coeffs


def fit_rotations(tau, quats_wxyz, degree):
    R = Rotation.from_quat(_wxyz2xyzw(quats_wxyz))
    R0 = R[0]
    # unwrap: cumulative frame-to-frame tangent steps (avoids the rotvec pi-wrap),
    # so a constant angular velocity is exactly linear in tau.
    cont = np.zeros((len(tau), 3))
    cont[1:] = np.cumsum((R[:-1].inv() * R[1:]).as_rotvec(), axis=0)
    coeffs, err = fit_basis(tau, cont, degree)
    return _xyzw2wxyz(R0.as_quat())[0], coeffs, err


def reconstruct_rotations(tau, q0_wxyz, coeffs):
    """Reconstruct orientation by composing INCREMENTAL exponentials of the
    fitted cumulative-rotvec curve — R(τ) = R0 · ∏ exp(Δcont_k) — rather than
    R0 · exp(cont(τ)).

    The latter (the prior implementation) is exact only for a *fixed* rotation
    axis; for a tumbling body the axis varies, exp(Σδ) ≠ ∏exp(δ) by an
    O(commutator) (BCH) term, and that error is independent of the fit degree —
    it was the ~0.6 rad geodesic floor in the degree sweep. Composing the
    increments is the correct SO(3) integral: increments are parallel for a
    fixed axis (so it stays exact there), and for a varying axis the only
    residual is the Chebyshev fit error of `cont`, which *does* shrink with
    degree.
    """
    tau = np.asarray(tau, float)
    order = np.argsort(tau, kind="stable")           # integrate in time order
    tau_s = tau[order]
    cont = reconstruct(tau_s, coeffs)                # (N,3) cumulative body rotvec
    cont0 = reconstruct(np.zeros(1), coeffs)[0]      # cont at τ=0, where R0 lives
    R = (Rotation.from_quat(_wxyz2xyzw(q0_wxyz)[0])
         * Rotation.from_rotvec(cont[0] - cont0))    # carry R0 to the first τ
    quats = np.empty((len(tau_s), 4))
    quats[0] = _xyzw2wxyz(R.as_quat())[0]
    for k, d in enumerate(np.diff(cont, axis=0)):    # right-multiply body-frame steps
        R = R * Rotation.from_rotvec(d)
        quats[k + 1] = _xyzw2wxyz(R.as_quat())[0]
    out = np.empty_like(quats)
    out[order] = quats                               # restore caller's τ order
    return out
