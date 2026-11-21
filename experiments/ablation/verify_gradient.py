"""
Verify the adjoint gradient of the steady-pressure forward model against
central finite differences on a small grid. Run before trusting any MCMC
result.

Usage: python experiments/ablation/verify_gradient.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ablation.forward import SteadyPressure


def make_observations(model, x_true, q=None, frac=0.2, noise_frac=0.05, seed=0):
    """Generate synthetic observations: pressure at `frac` of cells + noise."""
    rng = np.random.default_rng(seed)
    p_true = model.solve(x_true, q=q)
    n = model.n
    n_obs = max(1, int(round(frac * n)))
    obs_idx = rng.choice(n, n_obs, replace=False)
    sigma = noise_frac * np.std(p_true)
    y = p_true[obs_idx] + rng.normal(0.0, sigma, n_obs)
    return obs_idx, y, sigma


def log_lik(model, x, y_obs, obs_idx, sigma, q=None):
    p = model.solve(x, q=q)
    r = p[obs_idx] - y_obs
    return -0.5 * np.sum(r ** 2) / sigma ** 2


def main():
    nx, ny = 8, 5
    model = SteadyPressure(nx, ny)
    rng = np.random.default_rng(1)
    x_true = rng.normal(0.0, 0.5, nx * ny)   # log-perm field

    # well configuration: injector center, producer corner (unit rates)
    q = np.zeros(model.n)
    q[(nx // 2) * ny + (ny // 2)] = +1.0
    q[0] = -0.25
    q[ny - 1] = -0.25
    q[(nx - 1) * ny] = -0.25
    q[(nx - 1) * ny + (ny - 1)] = -0.25

    obs_idx, y_obs, sigma = make_observations(model, x_true, q=q, seed=2)

    grad_adj, ll = model.likelihood_grad(x_true, y_obs, obs_idx, sigma, q=q)
    print(f"log-lik = {ll:.6f}, sigma = {sigma:.6f}")

    # finite-difference check (Richardson extrapolation to remove O(h^2) error)
    n_check = 6
    idx = rng.choice(model.n, n_check, replace=False)
    h = 1e-5
    max_rel = 0.0
    print("\n  idx    adjoint          finite-diff(Rich)  rel.err")
    for i in idx:
        xp = x_true.copy(); xp[i] += h
        xm = x_true.copy(); xm[i] -= h
        f1 = (log_lik(model, xp, y_obs, obs_idx, sigma, q=q)
              - log_lik(model, xm, y_obs, obs_idx, sigma, q=q)) / (2 * h)
        xp = x_true.copy(); xp[i] += h / 2
        xm = x_true.copy(); xm[i] -= h / 2
        f2 = (log_lik(model, xp, y_obs, obs_idx, sigma, q=q)
              - log_lik(model, xm, y_obs, obs_idx, sigma, q=q)) / h
        fd = (4 * f2 - f1) / 3.0   # Richardson
        rel = abs(grad_adj[i] - fd) / max(1e-12, abs(fd))
        max_rel = max(max_rel, rel)
        print(f"  {i:4d}  {grad_adj[i]:+.8f}  {fd:+.8f}  {rel:.2e}")
    print(f"\nmax relative error = {max_rel:.2e}")
    print("PASS" if max_rel < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
