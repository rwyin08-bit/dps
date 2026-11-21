"""
Steady-state pressure forward model and adjoint gradient.

Forward problem:  div(K grad p) = q   (steady-state, single-phase pressure)
Discretized with a 5-point finite-volume stencil, harmonic-mean
transmissibility, and no-flux (Neumann) boundary conditions.

Parameterization: x = log(permeability), K = exp(x). This keeps K > 0
and matches the Gaussian-process prior on x.

The forward solve and the adjoint solve share one LU factorization of the
(symmetric) stiffness matrix, so each MALA step costs one factorization
plus two triangular solves.

Verified against finite differences in verify_gradient.py.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class SteadyPressure:
    """Steady-state pressure forward model with analytic adjoint gradient."""

    def __init__(self, nx, ny, dx=30.0, dy=30.0, h=10.0, reg=1e-8):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.dx = dx
        self.dy = dy
        self.h = h
        # geometric transmissibility factors (per unit mobility)
        self.gx = dy * h / dx   # x-direction edge factor
        self.gy = dx * h / dy   # y-direction edge factor
        self.reg = reg          # relative diagonal regularizer (pins the null space)
        self._edges = self._build_edge_list()

    def _build_edge_list(self):
        """Return list of interior edges: (i, j, geometric_factor).

        i < j are flat cell indices. The harmonic-mean transmissibility
        T_ij = g * 2 K_i K_j / (K_i + K_j).
        """
        nx, ny = self.nx, self.ny
        edges = []
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                if i < nx - 1:
                    edges.append((idx, idx + ny, self.gx))   # +x neighbor
                if j < ny - 1:
                    edges.append((idx, idx + 1, self.gy))    # +y neighbor
        return edges

    def _stiffness(self, Kf):
        """Build the symmetric stiffness matrix A(K) for flat K (length n)."""
        data, r_idx, c_idx = [], [], []
        d = np.zeros(self.n)
        for (i, j, g) in self._edges:
            Ki = Kf[i]; Kj = Kf[j]
            t = g * 2.0 * Ki * Kj / (Ki + Kj)   # harmonic-mean transmissibility
            data.append(-t); r_idx.append(i); c_idx.append(j)
            data.append(-t); r_idx.append(j); c_idx.append(i)
            d[i] += t; d[j] += t
        # diagonal regularizer pins the constant-pressure null space (Neumann);
        # scaled relative to the mean diagonal so it is robust across grids
        reg_abs = self.reg * (d.mean() + 1e-12)
        data += list(d + reg_abs)
        r_idx += list(np.arange(self.n))
        c_idx += list(np.arange(self.n))
        A = csr_matrix((data, (r_idx, c_idx)), shape=(self.n, self.n))
        return A

    def solve(self, x, q=None):
        """Forward solve: return pressure field p (flat) for log-perm x (flat)."""
        Kf = np.exp(np.asarray(x, dtype=float).ravel())
        A = self._stiffness(Kf)
        if q is None:
            q = np.zeros(self.n)
        b = np.asarray(q, dtype=float).ravel()
        lu = splu(A.tocsc())
        p = lu.solve(b)
        return p

    def _forward_and_factor(self, x, q):
        xa = np.asarray(x, dtype=float).ravel()
        # reject unphysical proposals (exp overflow) before they break the solver
        if not np.all(np.isfinite(xa)) or xa.max() > 30.0 or xa.min() < -30.0:
            return None, None, None, None
        Kf = np.exp(xa)
        A = self._stiffness(Kf)
        if q is None:
            q = np.zeros(self.n)
        b = np.asarray(q, dtype=float).ravel()
        try:
            lu = splu(A.tocsc())
            p = lu.solve(b)
        except RuntimeError:
            return None, None, None, None
        return Kf, A, lu, p

    def likelihood_grad(self, x, y_obs, obs_idx, sigma, q=None):
        """Negative log-likelihood gradient and value.

        Likelihood: y_obs ~ N(p[obs_idx], sigma^2 I), observation noise iid.

        Returns (grad_x, log_lik) where grad_x = d/dx log p(y|x).
        """
        Kf, A, lu, p = self._forward_and_factor(x, q)
        if lu is None:
            return np.full(self.n, np.nan), -np.inf
        residual = p[obs_idx] - np.asarray(y_obs).ravel()
        log_lik = -0.5 * np.sum(residual ** 2) / sigma ** 2

        # adjoint source: H^T r / sigma^2, nonzero only at observed cells
        adj_source = np.zeros(self.n)
        adj_source[obs_idx] = residual / sigma ** 2
        grad_loglik = self._gradient_from_adjoint(Kf, p, lu, adj_source)
        return grad_loglik, log_lik

    def _gradient_from_adjoint(self, Kf, p, lu, adj_source):
        """Given a factored forward solve and an adjoint source, return
        grad_x log p = -grad_x(0.5 ||r||^2 / sigma^2) for that source.

        adj_source is the vector r/sigma^2 scattered onto the grid (already
        including any output-space linear filter and 1/sigma^2).
        """
        lam = lu.solve(adj_source)          # A symmetric: A lam = adj_source
        grad = np.zeros(self.n)
        for (i, j, g) in self._edges:
            Ki = Kf[i]; Kj = Kf[j]
            dTdi = g * 2.0 * Kj * Kj / (Ki + Kj) ** 2 * Ki   # dT/dx_i
            dTdj = g * 2.0 * Ki * Ki / (Ki + Kj) ** 2 * Kj   # dT/dx_j
            D = (p[i] - p[j]) * (lam[i] - lam[j])
            grad[i] -= dTdi * D
            grad[j] -= dTdj * D
        return -grad
