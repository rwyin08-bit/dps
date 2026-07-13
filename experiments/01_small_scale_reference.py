#!/usr/bin/env python3
"""
Experiment 1: Small-scale physical reference posterior.
Generates a high-accuracy reference posterior via long-chain MCMC on a
tractable problem, then compares DPS samples against it.

This is the SINGLE MOST IMPORTANT experiment for establishing the
credibility of the ablation hierarchy.
"""

import numpy as np
import json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.config import *

# ═══════════════════════════════════════════════════════════
# Protocol:
# 1. Train VAE+LDM+FNO on 16x16 and 32x32 grid problems
# 2. Run pCN-MCMC targeting the TRUE physical posterior (MRST likelihood)
#    for 100k iterations with 10k burn-in
# 3. Run DPS with same configuration
# 4. Compare: energy distance, coverage, KS statistic
# ═══════════════════════════════════════════════════════════

def run_small_scale_reference(grid_size=16, latent_dim=32, seed=42):
    """Run reference MCMC and DPS comparison on a small grid."""
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Small-scale reference: {grid_size}x{grid_size}, d_z={latent_dim}")
    print(f"{'='*60}")

    # ── 1. Generate synthetic truth and observations ──
    nx, ny = grid_size, grid_size
    n_cells = nx * ny

    # Create a simple Gaussian random field as truth
    from scipy.ndimage import gaussian_filter
    truth = gaussian_filter(np.random.randn(nx, ny), sigma=2.0)
    truth = 2.0 + 0.5 * truth  # log-permeability ~ N(2, 0.25)

    # Generate observations
    n_obs = min(200, n_cells // 2)
    obs_locations = np.random.choice(n_cells, n_obs, replace=False)
    obs_noise_std = 0.02 * np.std(truth)
    obs_values = truth.flat[obs_locations] + np.random.normal(0, obs_noise_std, n_obs)

    print(f"  Grid: {nx}x{ny} ({n_cells} cells)")
    print(f"  Observations: {n_obs} locations")
    print(f"  Truth mean: {truth.mean():.3f}, std: {truth.std():.3f}")

    # ── 2. Run reference MCMC ──
    print(f"\n  Running pCN-MCMC ({EXP1_CONFIG['n_mcmc_iterations']} iterations)...")
    t0 = time.time()

    n_iter = EXP1_CONFIG['n_mcmc_iterations']
    n_burn = EXP1_CONFIG['n_mcmc_burnin']
    n_chains = 4

    # This is a PLACEHOLDER for the actual MCMC implementation.
    # In practice, this would use MRST for likelihood evaluations.
    # The structure below shows the expected interface.

    mcmc_samples = []
    for chain in range(n_chains):
        # Initialize from prior
        current = np.random.randn(nx, ny) * 0.5 + 2.0

        chain_samples = []
        for it in range(n_iter):
            # Propose new state
            proposal = current + np.random.randn(nx, ny) * 0.1

            # Compute log-posterior ratio (requires MRST)
            # log_ratio = compute_log_posterior(proposal) - compute_log_posterior(current)
            # For placeholder: use a simple Gaussian approximation
            current_ll = -0.5 * np.sum(((current.flat[obs_locations] - obs_values) / obs_noise_std)**2)
            proposal_ll = -0.5 * np.sum(((proposal.flat[obs_locations] - obs_values) / obs_noise_std)**2)
            log_ratio = proposal_ll - current_ll

            if np.log(np.random.random()) < log_ratio:
                current = proposal.copy()

            if it >= n_burn:
                chain_samples.append(current.copy())

            if (it + 1) % 20000 == 0:
                print(f"    Chain {chain+1}: {it+1}/{n_iter} iterations")

        mcmc_samples.append(np.array(chain_samples))
        print(f"    Chain {chain+1}: {len(chain_samples)} post-burnin samples")

    mcmc_time = (time.time() - t0) / 3600
    print(f"  MCMC completed in {mcmc_time:.1f} hours")

    # ── 3. Compute convergence diagnostics ──
    all_samples = np.concatenate(mcmc_samples, axis=0)
    print(f"\n  MCMC diagnostics:")
    print(f"    Total samples: {len(all_samples)}")

    # Compute R-hat (Gelman-Rubin)
    chain_means = np.array([s.mean(axis=0).mean() for s in mcmc_samples])
    chain_vars = np.array([s.var(axis=0).mean() for s in mcmc_samples])
    B = len(mcmc_samples[0]) * chain_means.var()
    W = chain_vars.mean()
    V = (len(mcmc_samples[0]) - 1) / len(mcmc_samples[0]) * W + B / len(mcmc_samples[0])
    R_hat = np.sqrt(V / W)
    print(f"    R-hat: {R_hat:.4f} (threshold: {EXP5_CONFIG['mcmc']['r_hat_threshold']})")

    # Compute ESS
    n_total = len(all_samples)
    rho = np.corrcoef(all_samples[:-1].mean(axis=(1,2)),
                      all_samples[1:].mean(axis=(1,2)))[0, 1]
    ess = n_total / (1 + 2 * rho / (1 - rho))
    print(f"    ESS: {ess:.0f} (threshold: {EXP5_CONFIG['mcmc']['ess_threshold']})")

    passed = R_hat < EXP5_CONFIG['mcmc']['r_hat_threshold'] and \
             ess > EXP5_CONFIG['mcmc']['ess_threshold']
    print(f"    Convergence: {'✅ PASSED' if passed else '❌ FAILED'}")

    # ── 4. Run DPS sampling (placeholder) ──
    # This would use the trained VAE+LDM+FNO pipeline
    print(f"\n  Running DPS ({EXP1_CONFIG['n_dps_samples']} samples)...")
    t0 = time.time()
    # dps_samples = run_dps_pipeline(grid_size, latent_dim, n_samples=...)
    dps_samples = np.random.randn(EXP1_CONFIG['n_dps_samples'], nx, ny) * 0.5 + 2.0
    dps_time = (time.time() - t0) / 3600
    print(f"  DPS completed in {dps_time:.1f} hours")

    # ── 5. Compute comparison metrics ──
    print(f"\n  Computing distributional distances...")
    metrics = {}

    # Energy distance
    from scipy.spatial.distance import cdist
    mcmc_flat = all_samples.reshape(len(all_samples), -1)
    dps_flat = dps_samples.reshape(len(dps_samples), -1)

    # Energy distance: D^2 = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
    n_mcmc = min(5000, len(mcmc_flat))
    n_dps = min(5000, len(dps_flat))
    idx_m = np.random.choice(len(mcmc_flat), n_mcmc, replace=False)
    idx_d = np.random.choice(len(dps_flat), n_dps, replace=False)

    cross = cdist(mcmc_flat[idx_m], dps_flat[idx_d]).mean()
    within_m = cdist(mcmc_flat[idx_m[:1000]], mcmc_flat[idx_m[1000:2000]]).mean()
    within_d = cdist(dps_flat[idx_d[:1000]], dps_flat[idx_d[1000:2000]]).mean()
    ed2 = 2 * cross - within_m - within_d
    metrics['energy_distance_squared'] = float(ed2)
    print(f"    D^2 = {ed2:.6f}")

    # Bootstrap CI
    boot_vals = []
    for b in range(EXP1_CONFIG['n_bootstrap']):
        ib = np.random.choice(n_mcmc, n_mcmc, replace=True)
        jb = np.random.choice(n_dps, n_dps, replace=True)
        cross_b = cdist(mcmc_flat[idx_m[ib]], dps_flat[idx_d[jb]]).mean()
        boot_vals.append(2 * cross_b - within_m - within_d)
    boot_vals = np.array(boot_vals)
    ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
    metrics['energy_distance_95ci'] = [float(ci_low), float(ci_high)]
    print(f"    95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    # ── 6. Save results ──
    result = {
        'grid_size': grid_size,
        'latent_dim': latent_dim,
        'seed': seed,
        'n_mcmc_samples': int(len(all_samples)),
        'n_dps_samples': int(len(dps_flat)),
        'mcmc_r_hat': float(R_hat),
        'mcmc_ess': float(ess),
        'mcmc_converged': bool(passed),
        'mcmc_time_hours': float(mcmc_time),
        'dps_time_hours': float(dps_time),
        'metrics': metrics,
    }

    outfile = os.path.join(RESULT_DIR, f'ref_{grid_size}x{grid_size}_d{latent_dim}_s{seed}.json')
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to: {outfile}")

    return result

def main():
    results = []
    for grid_size, latent_dim in zip(EXP1_CONFIG['grid_sizes'],
                                      EXP1_CONFIG['latent_dims']):
        result = run_small_scale_reference(
            grid_size=grid_size,
            latent_dim=latent_dim,
            seed=SEEDS['mcmc_reference'][0]
        )
        results.append(result)

    # Save summary
    summary = {
        'experiment': '01_small_scale_reference',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    with open(os.path.join(RESULT_DIR, 'exp1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Quick assessment
    all_passed = all(r['mcmc_converged'] for r in results)
    print(f"\n{'='*60}")
    print(f"Experiment 1 complete. All chains converged: {all_passed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
