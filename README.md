# Conditional Ablation of Surrogate Error in Bayesian Inversion

Reproducible code and data for the paper **"Conditional Ablation of Surrogate Error in Bayesian Inversion"** (submitted to *Engineering Computations*).

## Overview

This repository implements a conditional-ablation framework that isolates how a single surrogate property, the spectral attenuation of the forward operator, drives posterior degradation in gradient-based Bayesian inversion. A controlled frequency-attenuated surrogate with an exactly known roll-off is applied to steady-state pressure inversion on the SPE10 permeability benchmark, and the resulting posterior is compared to the reference posterior at the distribution, gradient, and spectral levels.

## Diagnostics

- **Energy distance** between posterior ensembles (distribution level)
- **Bandwise gradient cosine similarity** (gradient/spectral level)
- **Cross-spectral Wiener calibration** (proposed correction method)

## Repository structure

```
experiments/
├── ablation/
│   ├── forward.py          # steady-state pressure forward model + discrete adjoint
│   ├── surrogate.py        # frequency-attenuated (parameter-field) surrogate
│   ├── prior.py            # Matern-3/2 Gaussian-process prior (FFT/whitening)
│   ├── mala.py             # whitened Metropolis-adjusted Langevin sampler
│   ├── diagnostics.py      # energy distance, bandwise cosine, R-hat/ESS, Wiener filter
│   ├── calibrate.py        # cross-spectral Wiener calibration
│   ├── run.py              # end-to-end ablation experiment
│   ├── _run_identity.py    # identity-filter control (FFT round-trip)
│   ├── verify_gradient.py  # gradient check against automatic differentiation
│   ├── make_figures.py     # paper figures
│   ├── RECONCILIATION.md   # reported-number -> script/output mapping
│   ├── results/            # experiment outputs (JSON summaries)
│   └── figures/            # paper figures
└── data/
    └── spe10_perm_real.npy # SPE10 Model 2 permeability input
```

## Reproducing the experiments

```bash
pip install -r requirements.txt

# Ablation at both grid resolutions (whitened MALA, 3 chains)
python experiments/ablation/run.py --nx 8  --ny 5  --burnin 8000 --iter 6000 --chains 3 --thin 5
python experiments/ablation/run.py --nx 16 --ny 11 --burnin 8000 --iter 6000 --chains 3 --thin 5

# Identity-filter control (FFT round-trip, no attenuation)
python experiments/ablation/_run_identity.py

# Wiener calibration (per attenuation level)
python experiments/ablation/calibrate.py --nx 8 --ny 5 --alpha 2
python experiments/ablation/calibrate.py --nx 8 --ny 5 --alpha 4
python experiments/ablation/calibrate.py --nx 8 --ny 5 --alpha 8

# Paper figures
python experiments/ablation/make_figures.py
```

Every number reported in the paper maps to a script and output file; see `experiments/ablation/RECONCILIATION.md`.

## Requirements

Python 3.10+, numpy, scipy, matplotlib (see `requirements.txt`). The forward and surrogate gradients are verified against independent automatic-differentiation references in `verify_gradient.py`.

## License

MIT. See [LICENSE](LICENSE).
