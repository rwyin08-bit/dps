#!/usr/bin/env python3
"""
Experiment 2: Two-factor interaction (FNO accuracy x Tweedie approximation).
2x2 factorial design at 32x32 resolution with five independent replicates.

Hypothesis: interaction term Δ_{FNO×Tweedie} is non-zero.
"""

import numpy as np
import json, os, time, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.config import *

GRID = EXP2_CONFIG['grid_size']
DZ = EXP2_CONFIG['latent_dim']
N_TRAJ = EXP2_CONFIG['n_trajectories_per_cell']

def compute_energy_distance(samples_a, samples_b, max_samples=2000):
    """Compute energy distance between two sample sets."""
    from scipy.spatial.distance import cdist
    n_a = min(max_samples, len(samples_a))
    n_b = min(max_samples, len(samples_b))
    if n_a < 10 or n_b < 10:
        return np.nan, [np.nan, np.nan]

    idx_a = np.random.choice(len(samples_a), n_a, replace=False)
    idx_b = np.random.choice(len(samples_b), n_b, replace=False)
    a = samples_a[idx_a].reshape(n_a, -1)
    b = samples_b[idx_b].reshape(n_b, -1)

    cross = cdist(a, b).mean()
    within_a = cdist(a[:500], a[500:1000]).mean() if n_a >= 1000 else 0
    within_b = cdist(b[:500], b[500:1000]).mean() if n_b >= 1000 else 0
    d2 = 2 * cross - within_a - within_b

    # Bootstrap CI
    boots = []
    for _ in range(500):
        ia = np.random.choice(n_a, n_a, replace=True)
        ib = np.random.choice(n_b, n_b, replace=True)
        cb = cdist(a[ia], b[ib]).mean()
        boots.append(2 * cb - within_a - within_b)
    ci = np.percentile(boots, [2.5, 97.5])

    return float(d2), [float(ci[0]), float(ci[1])]

def run_cell(fno_level, tweedie_level, seed):
    """Run one cell of the 2x2 factorial design."""
    label = f"FNO={fno_level}, Tweedie={tweedie_level}"
    print(f"    Running: {label}")

    # ── This is a TEMPLATE for the actual implementation ──
    # In practice, this would:
    # 1. Train FNO with specified number of samples (or use MRST if 'exact')
    # 2. Configure DPS with specified Tweedie/SMC gradient
    # 3. Run N_TRAJ trajectories
    # 4. Compute metrics

    np.random.seed(seed)
    n_samples = min(N_TRAJ, 100)
    samples = np.random.randn(n_samples, GRID * GRID)
    return samples

def run_factorial(seed):
    """Run full 2x2 factorial for one replicate."""
    print(f"\n  Replicate (seed={seed}):")

    fno_keys = list(EXP2_CONFIG['fno_levels'].keys())
    twd_keys = list(EXP2_CONFIG['tweedie_levels'].keys())

    cell_results = {}
    for fkey, tkey in itertools.product(fno_keys, twd_keys):
        fno_val = EXP2_CONFIG['fno_levels'][fkey]
        twd_val = EXP2_CONFIG['tweedie_levels'][tkey]
        samples = run_cell(fno_val, twd_val, seed)
        cell_results[f'{fkey}_{tkey}'] = {
            'fno_level': fkey,
            'tweedie_level': tkey,
            'n_samples': len(samples),
        }

    # Compute interaction effect
    # Main effect of FNO: average over Tweedie levels
    # Main effect of Tweedie: average over FNO levels
    # Interaction: observed - (main_FNO + main_Tweedie + baseline)

    # For placeholder, report structure
    result = {
        'seed': seed,
        'cells': cell_results,
        'interaction_estimated': False,  # requires actual DPS runs
        'note': 'Placeholder structure - replace with actual DPS pipeline calls'
    }
    return result

def main():
    print(f"{'='*60}")
    print(f"Experiment 2: 2x2 Factorial Interaction")
    print(f"Grid: {GRID}x{GRID}, d_z={DZ}")
    print(f"{'='*60}")

    results = []
    for rep in range(min(3, N_REPLICATES)):  # start with 3 replicates
        result = run_factorial(SEEDS['dps_sample'][rep])
        results.append(result)

    summary = {
        'experiment': '02_interaction',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': EXP2_CONFIG,
        'results': results,
    }

    outfile = os.path.join(RESULT_DIR, 'exp2_summary.json')
    with open(outfile, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {outfile}")
    print(f"NOTE: This is a template. Replace placeholder sampling with actual DPS pipeline calls.")

if __name__ == '__main__':
    main()
