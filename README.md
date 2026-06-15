# DPS-HM: Surrogate-Guided Diffusion Posterior Sampling for Reservoir History Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

Official implementation. Paper: *Surrogate-Guided Diffusion Posterior Sampling for Reservoir History Matching*, R. Yin & S. Zhang.

## Abstract
DPS-HM couples a latent diffusion model with a differentiable FNO surrogate for Bayesian history matching. On SPE10 and Brugge: 2.4x data mismatch reduction over ES-MDA (p<0.01), diversity index 0.87.

## Installation
```bash
git clone https://github.com/rwyin08-bit/dps.git
cd dps
pip install -r requirements.txt
```
Requires Python 3.10+, PyTorch 2.1+, CUDA 12.1+.

## Quick Start
```bash
python scripts/train_vae.py --config configs/vae_spe10_2d.yaml
python scripts/train_ldm.py --config configs/ldm_spe10_2d.yaml
python scripts/train_fno.py --config configs/fno_spe10_2d.yaml
python scripts/sample_dps_hm.py --config configs/sampling_spe10_2d.yaml
python scripts/evaluate.py --results_dir ./results/
```

## Structure
| dir | purpose |
|-----|---------|
| src/vae/ | VAE encoder/decoder |
| src/ldm/ | Latent diffusion (U-Net, 280M) |
| src/fno/ | U-FNO surrogate |
| src/sampling/ | DPS-HM posterior sampling |
| src/utils/ | Metrics, visualization, I/O |
| configs/ | YAML hyperparameters |
| scripts/ | Training/evaluation scripts |
| notebooks/ | Figure reproduction |
| data/ | Dataset documentation |
| models/ | Pre-trained weights |

## Reproducibility
1. Download SPE10 from https://www.spe.org/web/csp/
2. Generate training data with MRST 2023b (scripts/mrst/)
3. Train VAE -> LDM -> FNO sequentially
4. Run DPS-HM sampling
5. Evaluate with scripts/evaluate.py

Pre-trained model weights: available upon reasonable request.

## Benchmarks
| Benchmark | Grid | Wells | Reference |
|-----------|------|-------|-----------|
| SPE10 2D | 220x60 | 5-spot | Christie & Blunt (2001) |
| SPE10 3D | 220x60x5 | 5-spot | Christie & Blunt (2001) |
| Brugge | 139x48x15 | 30 wells | Peters et al. (2010) |

## Data Availability
- SPE10: https://www.spe.org/web/csp/
- Brugge: ISAPP Brugge Field Study
- Generated datasets (10,000 realizations + 5,000 simulations): upon request

## Citation
```bibtex
@article{yin2025dpshm,
  title={Surrogate-Guided Diffusion Posterior Sampling for Reservoir History Matching},
  author={Yin, Rongwang and Zhang, Shaowei},
  year={2025}
}
```

## License
MIT. See [LICENSE](LICENSE).

## Contact
Corresponding author: Shaowei Zhang
