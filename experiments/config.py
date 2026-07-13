"""
Master configuration for Q1 validation experiments.
All parameters, paths, and random seeds are defined here.
"""

import os, json

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'data')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures_v3')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Random seeds (5 independent replicates)
# ═══════════════════════════════════════════════════════════
SEEDS = {
    'vae_train':    [42, 123, 456, 789, 555],
    'ldm_train':    [43, 124, 457, 790, 556],
    'fno_train':    [44, 125, 458, 791, 557],
    'dps_sample':   [45, 126, 459, 792, 558],
    'mcmc_reference': [46, 127, 460, 793, 559],
}
N_REPLICATES = len(SEEDS['vae_train'])

# ═══════════════════════════════════════════════════════════
# Experiment 1: Small-scale physical reference
# ═══════════════════════════════════════════════════════════
EXP1_CONFIG = {
    'description': 'Small-scale MCMC/SMC reference for DPS validation',
    'grid_sizes': [16, 32],        # 16x16 and 32x32 grids
    'latent_dims': [32, 64],       # corresponding latent dimensions
    'n_mcmc_iterations': 100_000,  # MCMC chain length
    'n_mcmc_burnin': 10_000,
    'n_smc_particles': 2000,       # SMC particles for gradient reference
    'n_dps_samples': 1000,         # DPS samples for comparison
    'metrics': ['energy_distance', 'wasserstein', 'mmd', 'ks_statistic',
                'coverage_80', 'coverage_90', 'coverage_95'],
    'n_bootstrap': 1000,           # bootstrap resamples for CI
}

# ═══════════════════════════════════════════════════════════
# Experiment 2: Two-factor interaction
# ═══════════════════════════════════════════════════════════
EXP2_CONFIG = {
    'description': '2x2 factorial: FNO accuracy x Tweedie approximation',
    'grid_size': 32,
    'latent_dim': 64,
    'fno_levels': {
        'low':   2500,   # FNO training samples
        'mid':   5000,
        'high':  10000,
        'exact': 'MRST',  # use full simulator
    },
    'tweedie_levels': {
        'point_estimate': 'Tweedie',  # standard DPS
        'smc_reference':  'SMC',      # SMC reference (2000 particles)
    },
    'n_trajectories_per_cell': 50,
    'ddim_steps': [250],
    'metrics': ['energy_distance', 'cosine_similarity', 'relative_l2'],
}

# ═══════════════════════════════════════════════════════════
# Experiment 3: Multi-seed statistics
# ═══════════════════════════════════════════════════════════
EXP3_CONFIG = {
    'description': 'Statistical robustness across independent training runs',
    'n_replicates': N_REPLICATES,
    'grid_size': 32,
    'latent_dim': 64,
    'n_dps_samples_per_replicate': 100,
    'report': ['mean', 'std', 'ci_95', 'min', 'max'],
}

# ═══════════════════════════════════════════════════════════
# Experiment 4: Posterior predictive checks
# ═══════════════════════════════════════════════════════════
EXP4_CONFIG = {
    'description': 'MRST validation of DPS posterior samples',
    'grid_size': 32,
    'latent_dim': 64,
    'n_validation_samples': 50,     # DPS samples to validate
    'n_mrst_per_sample': 1,         # MRST runs per sample
    'mrst_timeout_minutes': 30,     # timeout per simulation
    'observation_noise': 0.02,      # 2% noise
    'metrics': ['coverage', 'interval_width', 'predictive_log_likelihood',
                'rmse', 'mae'],
}

# ═══════════════════════════════════════════════════════════
# Experiment 5: Convergence diagnostics
# ═══════════════════════════════════════════════════════════
EXP5_CONFIG = {
    'description': 'Convergence diagnostics for all reference methods',
    'smc': {
        'n_particles_list': [500, 1000, 2000, 4000, 8000],
        'n_repeats': 5,
        'convergence_threshold': 0.01,  # relative change in ESS
    },
    'mcmc': {
        'n_chains': 4,
        'n_iterations': 100_000,
        'r_hat_threshold': 1.05,
        'ess_threshold': 200,
    },
    'ddim': {
        'step_list': [50, 100, 250, 500, 1000],
        'n_trajectories': 20,
        'eta_values': [0.0, 0.1, 0.5, 1.0],
    },
}

# ═══════════════════════════════════════════════════════════
# Experiment 6: OOD metrics
# ═══════════════════════════════════════════════════════════
EXP6_CONFIG = {
    'description': 'OOD quantification for FNO surrogate',
    'grid_size': 32,
    'latent_dim': 64,
    'n_dps_trajectory_samples': 500,  # samples along DPS trajectories
    'fno_training_samples': 4000,
    'ood_metrics': ['mmd', 'wasserstein', 'latent_mahalanobis',
                    'fno_error_correlation'],
    'n_bootstrap': 500,
}

# ═══════════════════════════════════════════════════════════
# Hardware configuration
# ═══════════════════════════════════════════════════════════
HARDWARE = {
    'gpu': 'NVIDIA A100 40GB',
    'cpu_cores': 64,
    'ram_gb': 256,
    'walltime_estimate_hours': {
        'exp1': 48,    # MCMC + SMC
        'exp2': 24,    # factorial ablation
        'exp3': 12,    # multi-seed
        'exp4': 72,    # MRST validation (most expensive)
        'exp5': 36,    # convergence
        'exp6': 8,     # OOD metrics
        'total': 200,  # ~1 week on single GPU + 64 CPU cores
    },
}

print("Configuration loaded successfully.")
print(f"  Experiment directory: {PROJECT_ROOT}")
print(f"  Replicates: {N_REPLICATES}")
print(f"  Estimated total walltime: ~{HARDWARE['walltime_estimate_hours']['total']} hours")
