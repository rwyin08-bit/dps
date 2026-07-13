# Q1 Validation Experiments for DPS Ablation Analysis

## Quick Start

```bash
# Dry run (plan without executing)
python experiments/run_all.py --dry-run

# Run all experiments (~200 hours total)
python experiments/run_all.py

# Run specific experiments
python experiments/run_all.py --exp 1        # Small-scale reference only
python experiments/run_all.py --exp 1,2,4    # Key experiments
```

## Experiment Overview

| # | Name | Walltime | Priority | Description |
|---|------|----------|----------|-------------|
| 1 | Small-scale reference | ~48 hr | 🔴 CRITICAL | MCMC targeting true physical posterior at 16×16 and 32×32 |
| 2 | Two-factor interaction | ~24 hr | 🔴 CRITICAL | FNO accuracy × Tweedie approximation factorial |
| 3 | Multi-seed statistics | ~12 hr | 🟡 HIGH | 5 independent training replicates |
| 4 | Posterior predictive checks | ~72 hr | 🟡 HIGH | MRST validation of DPS posterior samples |
| 5 | Convergence diagnostics | ~36 hr | 🟡 HIGH | SMC particle convergence, MCMC mixing, DDIM step convergence |
| 6 | OOD metrics | ~8 hr | 🟢 MEDIUM | MMD between FNO training and DPS trajectory samples |

## Prerequisites

1. **MRST (MATLAB Reservoir Simulation Toolbox)**: Download from https://www.sintef.no/projectweb/mrst/
2. **PyTorch** with CUDA support
3. **Pre-trained models**: VAE, LDM, FNO from the DPS-HM repository
4. **Data**: SPE10 benchmark, training realizations

## What Needs to be Implemented

The scripts in this directory are **templates** that define the experimental design, parameter configurations, output formats, and analysis protocols. They need to be connected to the actual DPS-HM codebase:

### Key integration points:

1. **`01_small_scale_reference.py`**: Replace the Gaussian placeholder likelihood with actual MRST calls. Implement pCN-MCMC with proper proposal tuning. Connect to the trained VAE+LDM+FNO pipeline.

2. **`02_interaction.py`**: Replace `run_cell()` placeholder with actual DPS pipeline calls. Implement SMC reference gradient computation.

3. **`04_predictive_checks.py`** (not yet written): Run MRST on DPS posterior samples. Compare production curves.

4. **`05_convergence.py`** (not yet written): Run SMC with varying particle counts. Monitor ESS and convergence.

5. **`06_ood_metrics.py`** (not yet written): Compute MMD between training and trajectory latent codes.

### Data flow:

```
config.py  →  run_all.py
                 ↓
          01_small_scale_reference.py  →  results/ref_*.json
          02_interaction.py            →  results/exp2_summary.json
          03_multiseed.py              →  results/exp3_summary.json
          04_predictive_checks.py      →  results/exp4_summary.json
          05_convergence.py            →  results/exp5_summary.json
          06_ood_metrics.py            →  results/exp6_summary.json
                 ↓
          analyze.py (to be written)   →  figures_v3/  (updated figures)
```

## Output

After running all experiments:

```bash
python experiments/analyze.py  # Generate updated figures and tables
```

This will produce updated versions of the 7 figures in `figures_v3/` with actual experimental data replacing the synthetic placeholders.
