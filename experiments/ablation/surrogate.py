"""
Frequency-attenuated surrogate G_alpha.

The surrogate attenuates the *parameter* field (log-permeability), not the
forward output. This models a surrogate that can only represent low-frequency
permeability structure (spectral truncation):

    G_alpha(x) = F( L_alpha(x) ),   L_alpha(x) = F^{-1}( H_alpha(|xi|) * F(x) )

where F is the forward model (steady pressure), F is the 2-D Fourier
transform, and H_alpha(|xi|) = exp( -|xi| / (alpha * xi0) ) is a radial
exponential low-pass filter with xi0 = 10.

The attenuation operator L_alpha is linear and self-adjoint, so the surrogate
likelihood gradient is the attenuated reference gradient evaluated at the
attenuated parameter:

    grad_x log p(y | G_alpha(x)) = L_alpha( grad_{x~} log p(y | x~) ),  x~ = L_alpha(x).
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
    """Apply the radial exponential low-pass filter to a 2-D field (real)."""
    F = np.fft.fft2(field2d)
    xi = radial_freq(field2d.shape[0], field2d.shape[1])
    H = np.exp(-xi / (alpha * xi0))
    return np.real(np.fft.ifft2(H * F))


class FrequencyAttenuatedSurrogate:
    """Parameter-attenuated surrogate with exactly known spectral roll-off.

    xi0 is the roll-off wavenumber. Default is ~Nyquist/5 (grid-relative),
    matching the paper's stated "one-fifth of the Nyquist frequency".
    """

    def __init__(self, forward_model, alpha, xi0=None):
        assert isinstance(forward_model, SteadyPressure)
        self.fwd = forward_model
        self.alpha = alpha
        self.nx = forward_model.nx
        self.ny = forward_model.ny
        if xi0 is None:
            xi0 = max(self.nx, self.ny) / 10.0   # ~ Nyquist / 5
        self.xi0 = xi0

    def attenuate_flat(self, field_flat):
        """Low-pass a flat field (log-perm or gradient), length n -> length n."""
        return attenuate(field_flat.reshape(self.nx, self.ny),
                         self.alpha, self.xi0).ravel()

    def likelihood_grad(self, x, y_obs, obs_idx, sigma, q=None):
        """Surrogate likelihood gradient and value.

        Likelihood: y_obs ~ N( F(L_alpha(x))[obs_idx], sigma^2 I ).
        Returns (grad_x, log_lik).
        """
        # attenuated parameter and its forward solve
        x_tilde = self.attenuate_flat(np.asarray(x).ravel())
        Kf, A, lu, p = self.fwd._forward_and_factor(x_tilde, q)
        residual = p[obs_idx] - np.asarray(y_obs).ravel()
        log_lik = -0.5 * np.sum(residual ** 2) / sigma ** 2

        # reference gradient at the attenuated parameter
        adj_source = np.zeros(self.fwd.n)
        adj_source[obs_idx] = residual / sigma ** 2
        g_x_tilde = self.fwd._gradient_from_adjoint(Kf, p, lu, adj_source)

        # chain rule through the (self-adjoint) low-pass filter
        grad_loglik = self.attenuate_flat(g_x_tilde)
        return grad_loglik, log_lik
