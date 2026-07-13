# DPS-HM: Surrogate-Guided Diffusion Posterior Sampling for Bayesian Reservoir History Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)]()

Official implementation of **DPS-HM**, a framework that couples a latent diffusion geological prior with a differentiable Fourier Neural Operator (FNO) surrogate for efficient Bayesian history matching under a score-based SDE conditioning perspective.

> **Paper**: Yin, R. & Zhang, S. "Surrogate-Guided Diffusion Posterior Sampling for Bayesian Reservoir History Matching" (2025)

## Highlights

- **Score-based SDE framework**: DPS-HM is grounded in the continuous-time reverse-time SDE formulation (Song et al., 2021), providing a principled theoretical foundation for the gradient-guided sampling procedure
- **Quantitative error analysis**: First quantitative characterization of the Tweedie approximation error in diffusion posterior sampling for PDE-constrained inverse problems
- **Posterior calibration (SBC)**: Simulation-based calibration validates that DPS-HM produces calibrated posterior samples (KS p=0.42, energy distance D²=0.012)
- **Spectral Entropy Effective Rank (SEER)**: DPS-HM achieves SEER=55.3 vs. 8.9–26.4 for ensemble baselines (out of 100 members)
- **40–50× online speedup**: 3 min for 100 posterior realizations vs. 200–220 hr for ensemble methods
- **6 baselines**: Compared against ES-MDA, LM-EnRML, PCA-ES-MDA, VAE-ES-MDA, and cINN-based inversion

## Installation

```bash
git clone https://github.com/rwyin08-bit/dps.git
cd dps
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.1+, CUDA 12.1+, NVIDIA GPU with ≥40 GB memory (A100 recommended)

## Quick Start

```bash
# 1. Train VAE for geological parameter compression
python scripts/train_vae.py --config configs/vae_spe10_2d.yaml

# 2. Train latent diffusion model (score network)
python scripts/train_ldm.py --config configs/ldm_spe10_2d.yaml

# 3. Train FNO surrogate
python scripts/train_fno.py --config configs/fno_spe10_2d.yaml

# 4. Run DPS-HM posterior sampling
python scripts/sample_dps_hm.py --config configs/sampling_spe10_2d.yaml

# 5. Evaluate results (including SBC diagnostics)
python scripts/evaluate.py --results_dir ./results/
```

## Repository Structure

```
dps/
├── src/
│   ├── vae/           # VAE encoder/decoder for parameter field compression
│   ├── ldm/           # Latent diffusion model (U-Net score network, ~280M params)
│   ├── fno/           # U-FNO surrogate with physics-informed mass-balance loss
│   ├── sampling/      # DPS-HM posterior sampler with adaptive step-size schedule
│   └── utils/         # Metrics (SEER, SBC, NRMSE, SSIM), visualization, I/O
├── configs/           # YAML hyperparameter configurations
├── scripts/           # Training, sampling, and evaluation scripts
├── data/              # Dataset documentation and download links
├── models/            # Pre-trained model weights (download instructions)
├── notebooks/         # Figure reproduction notebooks
└── scripts/mrst/      # MRST 2023b configuration (well schedules, fluid properties)
```

## Benchmarks

| Benchmark | Grid | Wells | Flow | Reference |
|-----------|------|-------|------|-----------|
| SPE10 2D | 220×60 (Layer 1) | 5-spot (1 inj + 4 prod) | 2-phase oil-water | Christie & Blunt (2001) |
| SPE10 3D | 220×60×5 (Layers 1–5) | 5-spot | 2-phase oil-water | Christie & Blunt (2001) |
| Brugge | 139×48×9 | 20 prod + 10 inj | 2-phase oil-water | Peters et al. (2010) |

> **Note**: SPE10 canonical grid is 60(NX)×220(NY)×85(NZ). We use 220×60 horizontal orientation (row-major transposition). Brugge uses the canonical 9-layer, two-phase specification.

## Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| VAE latent dim (2D) | 256 | Compression ratio ~52:1 per field |
| LDM parameters | ~280M | Modified U-Net with self-attention |
| FNO Fourier modes | 16/dim | 4 Fourier layers, 64 channels |
| DDIM sampling steps | 250 | Marginal gain beyond 250 steps <8% |
| Step size ζ₀, γ | 0.10, 0.98 | Adaptive schedule via running-statistic normalization |
| Ensemble size M | 100 | ~3 min on single A100 (40 GB) |
| DDIM stochasticity η | 0.1 | Enables MMD-based convergence assessment |

## Evaluation Metrics

| Dimension | Metrics |
|-----------|---------|
| Data match | NRMSE (WC, OR, BHP), DMR (95% PPI) |
| Geological plausibility | SSIM, Channel Connectivity, Variogram Match |
| Uncertainty quantification | PICP_90, Ensemble Spread, **SEER** (Spectral Entropy Effective Rank) |
| Posterior calibration | SBC rank histogram uniformity (KS test), Energy Distance |

## Reproducibility

All experiments use fixed random seeds:
- VAE: seed=42
- LDM: seed=123
- FNO: seed=456
- DPS sampling: seed=789

**Data sources**:
- SPE10: https://www.spe.org/web/csp/
- Brugge: ISAPP Brugge Field Study
- Pre-trained model weights: [Zenodo/Hugging Face] (or contact corresponding author)
- Generated datasets (10,000 realizations + 5,000 simulation pairs): upon request (>50 GB)

**MRST setup**: Configuration scripts for well schedules, fluid properties, and numerical solver settings are provided in `scripts/mrst/`. MRST 2023b is required for full-physics validation.

## Scope of Applicability

DPS-HM is designed for **fluvial-dominated clastic reservoirs** with available training images (brownfield redevelopment). Demonstrated on SPE10 (turbidite) and Brugge (shallow-marine Brent-type). Extension to carbonate, fractured, and unconventional reservoirs requires further research.

**Data requirements**: Minimum 500–1,000 geological training realizations + 500–1,000 full-physics simulations for surrogate training (fewer with transfer learning).

**Computational cost**:
- Offline training: ~58 GPU-hours (A100 40GB, one-time, amortized across campaigns)
- Online inference: ~3 min for 100 posterior realizations
- Break-even: 3–5 history matching campaigns vs. ES-MDA

## Citation

```bibtex
@article{yin2025dpshm,
  title={Surrogate-Guided Diffusion Posterior Sampling for Bayesian Reservoir History Matching},
  author={Yin, Rongwang and Zhang, Shaowei},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Contact

**Corresponding author**: Shaowei Zhang  
**Email**: swzhang189@163.com  
**Repository**: https://github.com/rwyin08-bit/dps
