"""
Generate paper figures from real ablation results.

Figures:
  1. D^2 vs attenuation level (8x5 and 16x11)
  2. gradient cosine vs attenuation level
  3. posterior mean fields (reference vs alpha=2,4,8)
  4. Wiener calibration gain and before/after alignment
"""

import numpy as np
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.labelsize": 10, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"ref": "#3C5488", "a2": "#E64B35", "a4": "#F39B7F", "a8": "#4DBBD5",
     "line": "#8491B4"}

alphas = [2, 4, 8]


def load_summary(grid):
    with open(os.path.join(RES, f"ablation_{grid}.json")) as f:
        return json.load(f)


# ---- Figure 1: D^2 vs attenuation ----
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for grid, ls, mk in [("8x5", "-", "o"), ("16x11", "--", "s")]:
    s = load_summary(grid)
    d2 = [s[f"alpha_{a}"]["D2_vs_reference"] for a in alphas]
    ax.plot(alphas, d2, ls, marker=mk, color=C["line"],
            label=f"grid {grid}")
ax.set_xlabel("Attenuation level $\\alpha$")
ax.set_ylabel("Energy distance $D^2$ vs reference")
ax.set_xticks(alphas)
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig1_d2.png")); plt.close(fig)

# ---- Figure 2: gradient cosine vs attenuation ----
fig, ax = plt.subplots(figsize=(3.4, 2.6))
s = load_summary("8x5")
cos = [s[f"alpha_{a}"]["gradient_cosine"] for a in alphas]
std = [s[f"alpha_{a}"].get("gradient_cosine_std", 0) for a in alphas]
ax.errorbar(alphas, cos, yerr=std, marker="o", color=C["ref"], capsize=3)
ax.set_xlabel("Attenuation level $\\alpha$")
ax.set_ylabel("Gradient cosine similarity")
ax.set_xticks(alphas)
ax.set_ylim(0.85, 1.01)
fig.savefig(os.path.join(FIG, "fig2_cosine.png")); plt.close(fig)

# ---- Figure 3: posterior mean fields ----
nx, ny = 8, 5
data = np.load(os.path.join(RES, "samples_8x5.npz"))
ref = data["reference"].reshape(-1, nx * ny)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
titles = ["reference", r"$\alpha=2$", r"$\alpha=4$", r"$\alpha=8$"]
for ax, (key, ttl) in zip(axes, [("reference", titles[0])] +
                          [(f"alpha_{a}", ttl) for a, ttl in zip(alphas, titles[1:])]):
    mean = data[key].reshape(-1, nx * ny).mean(axis=0).reshape(nx, ny)
    im = ax.imshow(mean, cmap="YlOrBr", aspect="auto")
    ax.set_title(ttl, fontsize=8); ax.axis("off")
fig.savefig(os.path.join(FIG, "fig3_fields.png")); plt.close(fig)

# ---- Figure 4: Wiener calibration ----
cal = {}
for a in alphas:
    with open(os.path.join(RES, f"calibration_8x5_a{a}.json")) as f:
        cal[a] = json.load(f)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(alphas, [cal[a]["cosine_before_mean"] for a in alphas], "-o",
        color=C["a2"], label="before")
ax.plot(alphas, [cal[a]["cosine_after_mean"] for a in alphas], "-s",
        color=C["ref"], label="after (Wiener)")
ax.set_xlabel("Attenuation level $\\alpha$")
ax.set_ylabel("Gradient cosine (held-out)")
ax.set_xticks(alphas)
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig4_calibration.png")); plt.close(fig)

print("Figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f)
