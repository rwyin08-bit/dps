"""
End-to-end conditional ablation experiment.

Reference posterior vs frequency-attenuated surrogate posteriors
(alpha in {2,4,8}), on the SPE10 permeability benchmark.

Usage:
    python experiments/ablation/run.py --nx 220 --ny 60 --burnin 5000 --iter 10000 --chains 5
"""

import numpy as np
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ablation.forward import SteadyPressure
from experiments.ablation.surrogate import FrequencyAttenuatedSurrogate
from experiments.ablation.prior import MaternGP
from experiments.ablation.mala import MALA, make_posterior
from experiments.ablation import diagnostics as diag


def load_spe10(nx, ny):
    """Load SPE10 Model 2 Layer 1 and down-sample to (nx, ny) log-permeability."""
    import scipy.ndimage as ndi
    perm = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "spe10_perm_real.npy"))
    zoom = (nx / perm.shape[0], ny / perm.shape[1])
    perm_ds = ndi.zoom(perm, zoom, order=1)
    return np.log(perm_ds + 1e-6)


def setup_wells(nx, ny):
    q = np.zeros(nx * ny)
    q[(nx // 2) * ny + (ny // 2)] = 1.0
    q[0] = -0.25; q[ny - 1] = -0.25
    q[(nx - 1) * ny] = -0.25; q[(nx - 1) * ny + (ny - 1)] = -0.25
    return q


def make_observations(model, x_true, q, frac=0.2, noise_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    p_true = model.solve(x_true, q=q)
    n_obs = max(1, int(round(frac * model.n)))
    obs_idx = rng.choice(model.n, n_obs, replace=False)
    sigma = noise_frac * np.std(p_true)
    y = p_true[obs_idx] + rng.normal(0.0, sigma, n_obs)
    return obs_idx, y, sigma


def run_config(model, prior, y_obs, obs_idx, sigma, q, x_init,
               n_burnin, n_iter, chains, thin, alpha=None, seed=0):
    """Run MALA for one configuration. Returns (chain_samples, summary)."""
    if alpha is None:
        lik_grad = lambda x: model.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)
        tag = "reference"
    else:
        sur = FrequencyAttenuatedSurrogate(model, alpha)
        lik_grad = lambda x: sur.likelihood_grad(x, y_obs, obs_idx, sigma, q=q)
        tag = f"alpha_{alpha}"

    post = make_posterior(lik_grad, prior)
    chain_samples = []
    accept_rates = []
    for c in range(chains):
        if c == 0:
            x0 = x_init
        else:
            x0 = prior.sample(np.random.default_rng(seed + 1000 * (c + 1)))
        m = MALA(post, x0, seed=seed + 1000 * (c + 1))
        samples, acc, _ = m.run(n_burnin, n_iter, thin=thin)
        chain_samples.append(samples)
        accept_rates.append(acc)

    rhat, ess = diag.monitor_rhat_ess(chain_samples)
    summary = {
        "tag": tag,
        "accept_rate": float(np.mean(accept_rates)),
        "r_hat": rhat,
        "ess": ess,
        "n_chains": chains,
        "n_samples_per_chain": int(len(chain_samples[0])),
    }
    return chain_samples, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=220)
    ap.add_argument("--ny", type=int, default=60)
    ap.add_argument("--burnin", type=int, default=5000)
    ap.add_argument("--iter", type=int, default=10000)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    x_true = load_spe10(nx, ny)
    model = SteadyPressure(nx, ny)
    q = setup_wells(nx, ny)

    mu = float(np.mean(x_true))
    sigma2 = float(np.var(x_true))
    rho = max(4.0, min(nx, ny) / 6.0)
    prior = MaternGP(nx, ny, sigma2=sigma2, rho=rho, mu=mu)

    obs_idx, y_obs, sigma = make_observations(model, x_true, q, seed=0)
    print(f"grid {nx}x{ny} (n={model.n}), prior mu={mu:.3f} sigma2={sigma2:.3f} rho={rho:.1f}")
    print(f"obs: {len(obs_idx)} cells, sigma={sigma:.5f}")

    configs = [None, 2, 4, 8]
    all_samples = {}
    summary = {}
    for alpha in configs:
        t0 = time.time()
        chains, s = run_config(model, prior, y_obs, obs_idx, sigma, q, x_true.copy(),
                               args.burnin, args.iter, args.chains, args.thin, alpha=alpha)
        s["wall_sec"] = time.time() - t0
        summary[s["tag"]] = s
        all_samples[s["tag"]] = np.stack(chains)   # (chains, n_samples, n)
        print(f"  {s['tag']:10s} accept={s['accept_rate']:.3f} "
              f"Rhat={s['r_hat']:.3f} ESS={s['ess']:.0f}  {s['wall_sec']:.1f}s")

    # energy distance: each surrogate posterior vs reference
    ref = all_samples["reference"].reshape(-1, model.n)
    for tag in ["alpha_2", "alpha_4", "alpha_8"]:
        sur = all_samples[tag].reshape(-1, model.n)
        d2 = diag.energy_distance(ref, sur, max_n=300)
        summary[tag]["D2_vs_reference"] = d2
        print(f"  {tag}: D^2 vs reference = {d2:.2f}")

    # gradient cosine at truth: reference gradient vs surrogate gradient
    g_star, _ = model.likelihood_grad(x_true, y_obs, obs_idx, sigma, q=q)
    for alpha, tag in [(2, "alpha_2"), (4, "alpha_4"), (8, "alpha_8")]:
        sur = FrequencyAttenuatedSurrogate(model, alpha)
        g_hat, _ = sur.likelihood_grad(x_true, y_obs, obs_idx, sigma, q=q)
        cos_b, scalar = diag.bandwise_cosine(g_star, g_hat, nx, ny)
        summary[tag]["gradient_cosine"] = scalar
        print(f"  {tag}: gradient cosine (at truth) = {scalar:.4f}")

    # save samples and summary
    np.savez_compressed(os.path.join(out_dir, f"samples_{nx}x{ny}.npz"),
                        **{k: v for k, v in all_samples.items()})
    out_path = args.out or os.path.join(out_dir, f"ablation_{nx}x{ny}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path} + samples_{nx}x{ny}.npz")


if __name__ == "__main__":
    main()
