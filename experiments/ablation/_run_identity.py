"""Identity-filter control: run the surrogate with H == 1 (FFT round-trip only)
and compare its posterior to the reference, on the 8x5 grid.

This isolates the FFT round-trip effect from the attenuation effect: if the
identity surrogate recovers the reference (D^2 ~ 0, cosine ~ 1), the FFT
round-trip itself introduces no bias.
"""
import numpy as np, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.ablation.forward import SteadyPressure
from experiments.ablation.surrogate import FrequencyAttenuatedSurrogate
from experiments.ablation.prior import MaternGP
from experiments.ablation import diagnostics as diag
from experiments.ablation import run as R

nx, ny = 8, 5
x_true = R.load_spe10(nx, ny)
model = SteadyPressure(nx, ny)
q = R.setup_wells(nx, ny)
prior = MaternGP(nx, ny, sigma2=float(np.var(x_true)),
                 rho=max(4.0, min(nx, ny) / 6.0), mu=float(np.mean(x_true)))
obs_idx, y_obs, sigma = R.make_observations(model, x_true, q, frac=0.3, noise_frac=0.2, seed=0)
print(f"grid {nx}x{ny}, sigma={sigma:.5f}")

# alpha = inf  ->  H = exp(-xi/(inf * xi0)) = exp(0) = 1  (pure FFT round-trip)
configs = [(None, "reference"), (float("inf"), "identity"), (2, "alpha_2"), (4, "alpha_4"), (8, "alpha_8")]
summary = {}
all_samples = {}
for alpha, tag in configs:
    t0 = time.time()
    chains, acc, rhat, ess = R.run_config(model, prior, alpha, y_obs, obs_idx, sigma, q,
                                          8000, 6000, 3, 5, 0.2)
    summary[tag] = {"accept_rate": acc, "r_hat": rhat, "ess": ess, "wall_sec": time.time() - t0}
    all_samples[tag] = np.stack(chains)
    print(f"  {tag:10s} accept={acc:.2f} Rhat={rhat:.2f} ESS={ess:.1f} {summary[tag]['wall_sec']:.0f}s")

ref = all_samples["reference"].reshape(-1, model.n)
for tag in ["identity", "alpha_2", "alpha_4", "alpha_8"]:
    s = all_samples[tag].reshape(-1, model.n)
    summary[tag]["D2_vs_reference"] = diag.energy_distance(ref, s, max_n=300)

for tag in ["identity", "alpha_2", "alpha_4", "alpha_8"]:
    a = 1e9 if tag == "identity" else int(tag.split("_")[1])
    sur_model = FrequencyAttenuatedSurrogate(model, a)
    cos_vals = []
    for s_i in range(30):
        xp = prior.sample(np.random.default_rng(s_i))
        g_star, _ = model.likelihood_grad(xp, y_obs, obs_idx, sigma, q=q)
        g_hat, _ = sur_model.likelihood_grad(xp, y_obs, obs_idx, sigma, q=q)
        _, scalar = diag.bandwise_cosine(g_star, g_hat, nx, ny)
        cos_vals.append(scalar)
    summary[tag]["gradient_cosine"] = float(np.mean(cos_vals))
    summary[tag]["gradient_cosine_std"] = float(np.std(cos_vals))

out = os.path.join(os.path.dirname(__file__), "results", "identity_control_8x5.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\nD2 vs reference:")
for tag in ["identity", "alpha_2", "alpha_4", "alpha_8"]:
    print(f"  {tag:10s} D2={summary[tag]['D2_vs_reference']:.4f}  cosine={summary[tag]['gradient_cosine']:.4f} (+/-{summary[tag]['gradient_cosine_std']:.4f})")
print(f"\nSaved {out}")
