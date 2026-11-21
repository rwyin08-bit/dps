"""
Frequency-attenuated surrogate G_alpha.

G_alpha(x) = F^{-1}( H_alpha(|xi|) * F( F(x) ) )

where F is the forward model (steady pressure), F is the 2-D Fourier
transform, and H_alpha(|xi|) = exp( -|xi| / (alpha * xi0) ) is a radial
exponential low-pass filter with xi0 = 10.

The attenuation operator L is linear and self-adjoint (H_alpha is real and
symmetric), so the surrogate likelihood gradient is computed with the same
adjoint machinery as the reference, with the residual attenuated first.
"""

import numpy as np
from .forward import SteadyPressure


def radial_freq(nx, ny):
    """Radial wavenumber |xi| in cycles-per-domain, shape (nx, ny)."""
    fx = np.fft.fftfreq(nx) * nx
    fy = np.fft.fftfreq(ny) * ny
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX ** 2 + FY ** 2)


def attenuate(field2d, alpha, xi0=10.0):
    """Apply the radial exponential low-pass filter to a 2-D field.

    Returns the real part of the filtered field (field is real-valued).
    """
    F = np.fft.fft2(field2d)
    xi = radial_freq(field2d.shape[0], field2d.shape[1])
    H = np.exp(-xi / (alpha * xi0))
    return np.real(np.fft.ifft2(H * F))


class FrequencyAttenuatedSurrogate:
    """Surrogate gradient operator with exactly known spectral roll-off."""

    def __init__(self, forward_model, alpha, xi0=10.0):
        assert isinstance(forward_model, SteadyPressure)
        self.fwd = forward_model
        self.alpha = alpha
        self.xi0 = xi0
        self.nx = forward_model.nx
        self.ny = forward_model.ny

    def attenuate_flat(self, p_flat):
        """Attenuate a flat pressure field (length n) -> flat."""
        return attenuate(p_flat.reshape(self.nx, self.ny), self.alpha, self.xi0).ravel()

    def likelihood_grad(self, x, y_obs, obs_idx, sigma, q=None):
        """Surrogate likelihood gradient and value.

        Likelihood: y_obs ~ N( G_alpha(x)[obs_idx], sigma^2 I ).
        Returns (grad_x, log_lik).
        """
        Kf, A, lu, p = self.fwd._forward_and_factor(x, q)
        p_hat = self.attenuate_flat(p)
        residual = p_hat[obs_idx] - np.asarray(y_obs).ravel()
        log_lik = -0.5 * np.sum(residual ** 2) / sigma ** 2

        # adjoint source: L^T H^T r / sigma^2, L self-adjoint
        ht_r = np.zeros(self.fwd.n)
        ht_r[obs_idx] = residual
        adj_source = self.attenuate_flat(ht_r) / sigma ** 2
        grad_loglik = self.fwd._gradient_from_adjoint(Kf, p, lu, adj_source)
        return grad_loglik, log_lik
