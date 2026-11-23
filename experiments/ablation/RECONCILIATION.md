# Reconciliation table: reported numbers to scripts and outputs

Every number reported in the revised manuscript traces to one script and one
output file in this directory. All experiments run on Python 3.10 with
numpy / scipy (requirements.txt at the repository root). Grids are down-sampled
SPE10 Model 2 Layer 1 permeability; observations are 30% of cells at 20% noise;
whitened MALA, 8,000 burn-in + 6,000 iterations thinned by 5, 3 independent
chains per configuration.

| Reported value | Script | Output file |
|---|---|---|
| Table I: acceptance rate, max R-hat, min ESS (8x5) | `run.py --nx 8 --ny 5 --burnin 8000 --iter 6000 --chains 3 --thin 5` | `results/ablation_8x5.json` |
| Table I: acceptance rate, max R-hat, min ESS (16x11) | `run.py --nx 16 --ny 11 --burnin 8000 --iter 6000 --chains 3 --thin 5` | `results/ablation_16x11.json` |
| Table II: D^2 (8x5) = 5.47 / 2.84 / 0.72 | `run.py` (energy_distance over saved chains) | `results/ablation_8x5.json` |
| Table II: D^2 (16x11) = 7.51 / 6.34 / 1.81 | `run.py` | `results/ablation_16x11.json` |
| Table II: D^2 bootstrap 95% CI (1,000 resamples) | bootstrap over `samples_*.npz` (cdist energy distance) | `results/samples_8x5.npz`, `results/samples_16x11.npz` |
| Table II: gradient cosine (8x5) = 0.91/0.97/0.99 (+/- std) | `run.py` (bandwise_cosine over 30 prior draws) | `results/ablation_8x5.json` |
| Table II: gradient cosine (16x11) = 0.89/0.98/1.00 (+/- std) | `run.py` | `results/ablation_16x11.json` |
| Identity control: D^2 = 0.13, cosine = 1.00 (8x5) | `_run_identity.py` | `results/identity_control_8x5.json` |
| Figures 1-3 | `make_figures.py` | `figures/fig{1,2,3}_*.png` |

The forward-model and surrogate gradients are checked against independent
automatic-differentiation references in `verify_gradient.py`.
