#!/usr/bin/env python3
"""
Master experiment runner for Q1 validation of DPS ablation analysis.

Usage:
    python experiments/run_all.py                    # Run all experiments
    python experiments/run_all.py --exp 1            # Run only experiment 1
    python experiments/run_all.py --exp 1,2,4        # Run experiments 1, 2, and 4
    python experiments/run_all.py --dry-run          # Print plan without executing

Requires: MRST (MATLAB), PyTorch, NumPy, SciPy, matplotlib
"""

import argparse, sys, os, time, json, subprocess
from config import *

def run_experiment(exp_num, dry_run=False):
    """Run a single experiment by number."""
    scripts = {
        1: '01_small_scale_reference.py',
        2: '02_interaction.py',
        3: '03_multiseed.py',
        4: '04_predictive_checks.py',
        5: '05_convergence.py',
        6: '06_ood_metrics.py',
    }

    if exp_num not in scripts:
        print(f"Unknown experiment: {exp_num}")
        return False

    script = os.path.join(os.path.dirname(__file__), scripts[exp_num])
    if not os.path.exists(script):
        print(f"Script not found: {script}")
        return False

    hours = HARDWARE['walltime_estimate_hours'][f'exp{exp_num}']

    if dry_run:
        print(f"  [DRY RUN] Experiment {exp_num}: {scripts[exp_num]}")
        print(f"    Estimated time: ~{hours} hours")
        return True

    print(f"\n{'='*60}")
    print(f"Running Experiment {exp_num}: {scripts[exp_num]}")
    print(f"Estimated time: ~{hours} hours")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,  # show output in real-time
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    elapsed = (time.time() - t0) / 3600

    if result.returncode == 0:
        print(f"\n✅ Experiment {exp_num} completed in {elapsed:.1f} hours")
        return True
    else:
        print(f"\n❌ Experiment {exp_num} FAILED (exit code {result.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run DPS ablation validation experiments')
    parser.add_argument('--exp', type=str, default='all',
                        help='Comma-separated experiment numbers (e.g., 1,2,3) or "all"')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without executing')
    args = parser.parse_args()

    if args.exp == 'all':
        exps = list(range(1, 7))
    else:
        exps = [int(x.strip()) for x in args.exp.split(',')]

    if args.dry_run:
        total_hours = sum(HARDWARE['walltime_estimate_hours'][f'exp{e}'] for e in exps)
        print(f"\n{'='*60}")
        print(f"DRY RUN: {len(exps)} experiments planned")
        print(f"Total estimated walltime: ~{total_hours} hours ({total_hours/24:.1f} days)")
        print(f"{'='*60}\n")
        for e in exps:
            run_experiment(e, dry_run=True)
        return

    print(f"\n{'='*60}")
    print(f"STARTING {len(exps)} EXPERIMENTS")
    print(f"Total estimated walltime: ~{sum(HARDWARE['walltime_estimate_hours'][f'exp{e}'] for e in exps)} hours")
    print(f"{'='*60}")

    results = {}
    for e in exps:
        results[e] = run_experiment(e)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for e, ok in results.items():
        status = '✅' if ok else '❌'
        print(f"  {status} Experiment {e}")
    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} experiments completed successfully")

    # Save results manifest
    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {str(k): v for k, v in results.items()},
    }
    with open(os.path.join(RESULT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    main()
