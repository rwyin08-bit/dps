"""
End-to-end conditional ablation experiment (real, reproducible).

Reference posterior vs parameter-attenuated surrogate posteriors
(alpha in {2,4,8}) on the SPE10 benchmark, using a whitened (preconditioned)
MALA sampler and reporting D^2, gradient cosine, R-hat, ESS, and acceptance.

Usage:
    python experiments/ablation/run.py --nx 64 --ny 44 --nf 0.2 --burnin 6000 --iter 6000 --chains 5
"""

import numpy as np
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ablation.forward import SteadyPressure
from experiments.ablation.surrogate import FrequencyAttenuatedSurrogate
from experiments.ablation.prior import MaternGP
from experiments.ablation.mala import MALA
from experiments.ablation import diagnostics as diag


def load_spe10(nx, ny):
    import scipy.ndimage as ndi
    perm = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "spe10_perm_real.npy"))
    perm_ds = ndi.zoom(perm, (nx / perm.shape[0], ny / perm.shape[1]), order=1)
    return np.log(perm_ds + 1e-6)


def setup_wells(nx, ny):
    q = np.zeros(nx * ny)
    q[(nx // 2) * ny + (ny // 2)] = 1.0
    q[0] = -0.25; q[ny - 1] = -0.25
    q[(nx - 1) * ny] = -0.25; q[(nx - 1) * ny + (ny - 1)] = -0.25
    return q


def make_observations(model, x_true, q, frac=0.3, noise_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    p_true = model.solve(x_true, q=q)
    n_obs = max(1, int(round(frac * model.n)))
    obs_idx = rng.choice(model.n, n_obs, replace=False)
    sigma = noise_frac * np.std(p_true)
    y = p_true[obs_idx] + rng.normal(0.0, sigma, n_obs)
    return obs_idx, y, sigma


def make_likelihood(model, alpha, y_obs, obs_idx, sigma, q):
    if alpha is None:
        return lambda x: model.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)
    sur = FrequencyAttenuatedSurrogate(model, alpha)
    return lambda x: sur.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)


def make_post_w(prior, lik):
    def post_w(w):
        x = prior.unwhiten(w)
        g_ll, ll = lik(x)
        g_w = prior.apply_Csqrt(g_ll) - w
        return ll - 0.5 * np.dot(w, w), g_w
    return post_w


def run_config(model, prior, alpha, y_obs, obs_idx, sigma, q,
               n_burnin, n_iter, chains, thin, step_size, seed=0):
    lik = make_likelihood(model, alpha, y_obs, obs_idx, sigma, q)
    post_w = make_post_w(prior, lik)
    rng = np.random.default_rng(seed)
    chain_samples = []
    accept_rates = []
    for c in range(chains):
        w0 = rng.standard_normal(model.n)
        m = MALA(post_w, w0, step_size=step_size, seed=seed + 1000 * (c + 1))
        s, acc, _ = m.run(n_burnin, n_iter, thin=thin)
        chain_samples.append(np.array([prior.unwhiten(si) for si in s]))
        accept_rates.append(acc)
    rhat, ess = diag.monitor_rhat_ess(chain_samples)
    return chain_samples, float(np.mean(accept_rates)), rhat, ess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=44)
    ap.add_argument("--nf", type=float, default=0.2)
    ap.add_argument("--burnin", type=int, default=6000)
    ap.add_argument("--iter", type=int, default=6000)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    x_true = load_spe10(nx, ny)
    model = SteadyPressure(nx, ny)
    q = setup_wells(nx, ny)
    prior = MaternGP(nx, ny, sigma2=float(np.var(x_true)),
                     rho=max(4.0, min(nx, ny) / 6.0), mu=float(np.mean(x_true)))
    obs_idx, y_obs, sigma = make_observations(model, x_true, q,
                                              noise_frac=args.nf, seed=0)
    print(f"grid {nx}x{ny} (n={model.n}), sigma={sigma:.5f}", flush=True)

    all_samples = {}
    summary = {}
    for alpha in [None, 2, 4, 8]:
        tag = "reference" if alpha is None else f"alpha_{alpha}"
        t0 = time.time()
        chains, acc, rhat, ess = run_config(
            model, prior, alpha, y_obs, obs_idx, sigma, q,
            args.burnin, args.iter, args.chains, args.thin, args.step)
        summary[tag] = {"accept_rate": acc, "r_hat": rhat, "ess": ess,
                        "wall_sec": time.time() - t0}
        all_samples[tag] = np.stack(chains)
        print(f"  {tag:10s} accept={acc:.2f} Rhat={rhat:.2f} ESS={ess:.0f} "
              f"{summary[tag]['wall_sec']:.0f}s", flush=True)

    # D^2 of each surrogate posterior vs reference
    ref = all_samples["reference"].reshape(-1, model.n)
    for alpha in [2, 4, 8]:
        tag = f"alpha_{alpha}"
        sur = all_samples[tag].reshape(-1, model.n)
        summary[tag]["D2_vs_reference"] = diag.energy_distance(ref, sur, max_n=300)

    # gradient cosine at prior samples (meaningful points, not the noise-dominated
    # posterior mean), averaged over several draws
    for alpha in [2, 4, 8]:
        sur_model = FrequencyAttenuatedSurrogate(model, alpha)
        cos_vals = []
        for s in range(30):
            xp = prior.sample(np.random.default_rng(s))
            g_star, _ = model.likelihood_grad(xp, y_obs, obs_idx, sigma, q=q)
            g_hat, _ = sur_model.likelihood_grad(xp, y_obs, obs_idx, sigma, q=q)
            _, scalar = diag.bandwise_cosine(g_star, g_hat, nx, ny)
            cos_vals.append(scalar)
        summary[f"alpha_{alpha}"]["gradient_cosine"] = float(np.mean(cos_vals))
        summary[f"alpha_{alpha}"]["gradient_cosine_std"] = float(np.std(cos_vals))

    np.savez_compressed(os.path.join(out_dir, f"samples_{nx}x{ny}.npz"),
                        **{k: v for k, v in all_samples.items()})
    out_path = args.out or os.path.join(out_dir, f"ablation_{nx}x{ny}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nD2 vs reference:", flush=True)
    for alpha in [2, 4, 8]:
        tag = f"alpha_{alpha}"
        print(f"  {tag}: D2={summary[tag]['D2_vs_reference']:.4f} "
              f"cosine={summary[tag]['gradient_cosine']:.4f}", flush=True)
    print(f"\nSaved {out_path} + samples_{nx}x{ny}.npz")


if __name__ == "__main__":
    main()
