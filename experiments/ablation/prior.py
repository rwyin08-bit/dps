"""
Gaussian-process prior with Matern-3/2 covariance on a regular grid.

Implemented in the Fourier domain: for a stationary GP on a regular grid
the covariance is (approximately) circulant, so C and C^{-1} act by FFT
multiplication. This keeps the prior gradient O(n log n) even at 220 x 60.

x = log(permeability), with mean mu and marginal variance sigma2.
"""

import numpy as np


class MaternGP:
    """2-D Matern-3/2 GP prior with spectral (FFT) covariance operators."""

    def __init__(self, nx, ny, sigma2=1.0, rho=8.0, mu=0.0):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.sigma2 = sigma2
        self.rho = rho
        self.mu = mu
        self._spec = self._spectrum()

    def _spectrum(self):
        fx = np.fft.fftfreq(self.nx)
        fy = np.fft.fftfreq(self.ny)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        k2 = FX ** 2 + FY ** 2
        # Matern-3/2 spectral density (d=2), k in cycles per cell
        const = self.sigma2 * 18.0 * np.sqrt(3.0) * np.pi / self.rho ** 3
        S = const * (3.0 / self.rho ** 2 + 4.0 * np.pi ** 2 * k2) ** (-2.5)
        # regularize the zero mode (constant shift has no likelihood effect)
        S = np.maximum(S, 1e-12)
        return S

    def _apply_Cinv(self, x_flat):
        x2 = (np.asarray(x_flat).ravel() - self.mu).reshape(self.nx, self.ny)
        return np.real(np.fft.ifft2(np.fft.fft2(x2) / self._spec)).ravel()

    def apply_Csqrt(self, v_flat):
        """C^{1/2} v (no mean shift)."""
        v = np.asarray(v_flat).ravel().reshape(self.nx, self.ny)
        return np.real(np.fft.ifft2(np.sqrt(self._spec) * np.fft.fft2(v))).ravel()

    def apply_Cinvsqrt(self, v_flat):
        """C^{-1/2} v (no mean shift)."""
        v = np.asarray(v_flat).ravel().reshape(self.nx, self.ny)
        return np.real(np.fft.ifft2(np.fft.fft2(v) / np.sqrt(self._spec))).ravel()

    def whiten(self, x_flat):
        """w = C^{-1/2}(x - mu) (standard-normal preconditioned coordinates)."""
        d = (np.asarray(x_flat).ravel() - self.mu).reshape(self.nx, self.ny)
        return np.real(np.fft.ifft2(np.fft.fft2(d) / np.sqrt(self._spec))).ravel()

    def unwhiten(self, w_flat):
        """x = mu + C^{1/2} w (inverse of whiten)."""
        w = np.asarray(w_flat).ravel().reshape(self.nx, self.ny)
        return (self.mu + np.real(np.fft.ifft2(np.sqrt(self._spec) * np.fft.fft2(w))).ravel())

    def log_p(self, x_flat):
        d = np.asarray(x_flat).ravel() - self.mu
        Cinv_d = self._apply_Cinv(d)
        return -0.5 * np.dot(d, Cinv_d)

    def grad(self, x_flat):
        """grad_x log p(x) = -C^{-1}(x - mu)."""
        return -self._apply_Cinv(np.asarray(x_flat).ravel())

    def sample(self, rng):
        """Draw one field from the prior (via standard-normal whitening)."""
        w = rng.standard_normal(self.nx * self.ny)
        return self.unwhiten(w)
