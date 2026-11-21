"""
Cross-spectral Wiener calibration with held-out validation.

Fits the Wiener filter C(k) = P_gh / (P_hh + lambda) from N_cal calibration
gradient pairs (reference g*, surrogate g^), then evaluates on held-out
points whether the calibrated gradient g~ = C g^ aligns better with g*
than the raw surrogate gradient does.

This is the empirical validation that responds to R1-3 / R2-3 without
running calibrated MALA chains.

Usage:
    python experiments/ablation/calibrate.py --nx 64 --ny 44 --nf 0.2 --ncal 500
"""

import numpy as np
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ablation.forward import SteadyPressure
from experiments.ablation.surrogate import FrequencyAttenuatedSurrogate
from experiments.ablation.prior import MaternGP
from experiments.ablation import diagnostics as diag
from experiments.ablation.run import load_spe10, setup_wells, make_observations


def grad_pair(model, prior, alpha, x, y_obs, obs_idx, sigma, q):
    g_star, _ = model.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)
    if alpha is None:
        return g_star, g_star.copy()
    sur = FrequencyAttenuatedSurrogate(model, alpha)
    g_hat, _ = sur.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)
    return g_star, g_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=44)
    ap.add_argument("--nf", type=float, default=0.2)
    ap.add_argument("--ncal", type=int, default=500)
    ap.add_argument("--nheld", type=int, default=100)
    ap.add_argument("--alpha", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    model = SteadyPressure(nx, ny)
    x_true = load_spe10(nx, ny)
    q = setup_wells(nx, ny)
    prior = MaternGP(nx, ny, sigma2=float(np.var(x_true)),
                     rho=max(4.0, min(nx, ny) / 6.0), mu=float(np.mean(x_true)))
    obs_idx, y_obs, sigma = make_observations(model, x_true, q, noise_frac=args.nf, seed=0)

    rng = np.random.default_rng(7)
    cal = [prior.sample(rng) for _ in range(args.ncal)]
    held = [prior.sample(rng) for _ in range(args.nheld)]

    alpha = args.alpha
    t0 = time.time()
    wc = diag.WienerCalibration(nx, ny)
    gs_list, gh_list = [], []
    for x in cal:
        gs, gh = grad_pair(model, prior, alpha, x, y_obs, obs_idx, sigma, q)
        gs_list.append(gs); gh_list.append(gh)
    wc.fit(gs_list, gh_list)
    fit_sec = time.time() - t0

    cos_before, cos_after = [], []
    for x in held:
        gs, gh = grad_pair(model, prior, alpha, x, y_obs, obs_idx, sigma, q)
        gt = wc.apply(gh)
        cos_before.append(diag.bandwise_cosine(gs, gh, nx, ny)[1])
        cos_after.append(diag.bandwise_cosine(gs, gt, nx, ny)[1])

    result = {
        "alpha": alpha,
        "n_cal": args.ncal,
        "n_held": args.nheld,
        "cosine_before_mean": float(np.mean(cos_before)),
        "cosine_before_std": float(np.std(cos_before)),
        "cosine_after_mean": float(np.mean(cos_after)),
        "cosine_after_std": float(np.std(cos_after)),
        "fit_sec": fit_sec,
    }
    print(f"Wiener calibration alpha={alpha} ({args.ncal} cal, {args.nheld} held):", flush=True)
    print(f"  gradient cosine  before: {result['cosine_before_mean']:.4f} +/- {result['cosine_before_std']:.4f}", flush=True)
    print(f"  gradient cosine  after : {result['cosine_after_mean']:.4f} +/- {result['cosine_after_std']:.4f}", flush=True)

    out_path = args.out or os.path.join(os.path.dirname(__file__), "results",
                                        f"calibration_{nx}x{ny}_a{alpha}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
