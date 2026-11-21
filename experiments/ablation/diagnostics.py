"""
Posterior diagnostics: energy distance, bandwise gradient cosine,
R-hat, ESS, and cross-spectral Wiener calibration.
"""

import numpy as np
from scipy.spatial.distance import cdist


def energy_distance(X, Y, max_n=300, rng=None):
    """Energy distance D^2 between two sample sets (rows = samples)."""
    X = np.asarray(X); Y = np.asarray(Y)
    if rng is None:
        rng = np.random.default_rng(0)
    n_a = min(max_n, len(X)); n_b = min(max_n, len(Y))
    if n_a < 2 or n_b < 2:
        return np.nan
    a = X[rng.choice(len(X), n_a, replace=False)]
    b = Y[rng.choice(len(Y), n_b, replace=False)]
    cross = cdist(a, b).mean()
    within_a = cdist(a, a)[np.triu_indices(n_a, 1)].mean()
    within_b = cdist(b, b)[np.triu_indices(n_b, 1)].mean()
    return float(2 * cross - within_a - within_b)


def bandwise_cosine(g_star, g_hat, nx, ny, n_bands=16, eps=1e-8):
    """Bandwise cosine similarity of two gradient fields, plus scalar summary.

    g_star, g_hat: flat gradient fields (length nx*ny).
    Returns (cos_b: array over bands, scalar_cos: energy-weighted average).
    """
    a = np.asarray(g_star).reshape(nx, ny)
    b = np.asarray(g_hat).reshape(nx, ny)
    A = np.fft.fft2(a); B = np.fft.fft2(b)
    fx = np.fft.fftfreq(nx) * nx
    fy = np.fft.fftfreq(ny) * ny
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    r = np.sqrt(FX ** 2 + FY ** 2)
    rmax = r.max()
    bands = np.linspace(0, rmax, n_bands + 1)

    cos_b = []
    weights = []
    for b in range(n_bands):
        mask = (r >= bands[b]) & (r < bands[b + 1])
        if mask.sum() == 0:
            continue
        num = np.real(np.sum(A[mask] * np.conj(B[mask])))
        den = np.sqrt(np.sum(np.abs(A[mask]) ** 2) * np.sum(np.abs(B[mask]) ** 2)) + eps
        cos_b.append(num / den)
        weights.append(np.sum(np.abs(A[mask]) ** 2))
    cos_b = np.array(cos_b)
    weights = np.array(weights)
    scalar = float(np.sum(weights * cos_b) / (np.sum(weights) + eps))
    return cos_b, scalar


def r_hat_per_component(chains):
    """Gelman-Rubin R-hat per parameter component (vectorized).

    chains: list of (n_samples, n_params) arrays.
    Returns an array of R-hat values, one per component.
    """
    arr = np.stack([np.asarray(c, dtype=float) for c in chains])  # (m, n, d)
    m, n, d = arr.shape
    if n < 2 or m < 2:
        return np.ones(d)
    cm = arr.mean(axis=1)           # (m, d)
    cv = arr.var(axis=1, ddof=1)    # (m, d)
    W = cv.mean(axis=0)             # (d,)
    B = n * cm.var(axis=0, ddof=1)  # (d,)
    V = (n - 1) / n * W + B / n
    return np.sqrt(V / np.maximum(W, 1e-12))


def ess_per_component(chain):
    """Effective sample size per component via Geyer's initial positive sequence.

    chain: (n_samples, n_params). Returns an array of ESS, one per component.
    """
    x = np.asarray(chain, dtype=float)
    n, d = x.shape
    if n < 2:
        return np.ones(d)
    x = x - x.mean(axis=0, keepdims=True)
    nfft = 1
    while nfft < 2 * n - 1:
        nfft *= 2
    F = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:n] / n  # (n, d)
    var = acov[0]
    out = np.ones(d)
    tau = np.ones(d)
    for k in range(1, (n - 1) // 2):
        pair = (acov[2 * k - 1] + acov[2 * k]) / np.maximum(var, 1e-12)
        if np.all(pair <= 0):
            break
        tau = tau + np.where(pair > 0, pair, 0.0)
    return n / np.maximum(tau, 1e-12)


def monitor_rhat_ess(samples_by_chain):
    """Max R-hat and min ESS across all parameter components."""
    chains = [np.asarray(c, dtype=float) for c in samples_by_chain]
    rhat = r_hat_per_component(chains)
    ess_arr = ess_per_component(np.concatenate(chains, axis=0))
    return float(np.max(rhat)), float(np.min(ess_arr))


class WienerCalibration:
    """Cross-spectral Wiener filter estimated from calibration gradient pairs."""

    def __init__(self, nx, ny, lam=1e-6):
        self.nx = nx; self.ny = ny; self.lam = lam
        self.P_gg = None
        self.P_hh = None
        self.P_gh = None
        self.C = None

    def fit(self, g_star_list, g_hat_list):
        """g_star_list, g_hat_list: lists of flat reference/surrogate gradients."""
        P_gg = P_hh = P_gh = 0.0
        for gs, gh in zip(g_star_list, g_hat_list):
            A = np.fft.fft2(np.asarray(gs).reshape(self.nx, self.ny))
            B = np.fft.fft2(np.asarray(gh).reshape(self.nx, self.ny))
            P_gg += np.abs(A) ** 2
            P_hh += np.abs(B) ** 2
            P_gh += A * np.conj(B)
        m = len(g_star_list)
        self.P_gg = P_gg / m
        self.P_hh = P_hh / m
        self.P_gh = P_gh / m
        self.C = self.P_gh / (self.P_hh + self.lam)
        return self

    def apply(self, g_hat_flat):
        """Apply the Wiener filter to a surrogate gradient -> calibrated gradient."""
        B = np.fft.fft2(np.asarray(g_hat_flat).reshape(self.nx, self.ny))
        return np.real(np.fft.ifft2(self.C * B)).ravel()
