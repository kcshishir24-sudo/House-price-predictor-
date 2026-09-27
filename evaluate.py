"""evaluate.py - produce Figure 4.3 (model comparison) and Figure 4.4 (actual vs predicted)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

meta = json.load(open("models/meta.json"))
res, best = meta["results"], meta["selected_model"]
names = list(res)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, key, title in zip(ax, ["test_r2", "mae_lakh", "mape_pct"], ["Test R²", "MAE (NPR Lakh)", "MAPE (%)"]):
    vals = [res[n][key] for n in names]
    bars = a.bar(range(len(names)), vals, color=["#b3261e" if n == best else "#9aa7b4" for n in names])
    a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    a.set_title(title)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Model performance comparison (selected: {best})"); fig.tight_layout()
fig.savefig("reports/fig_4_3_model_comparison.png", dpi=150)

y, p = np.load("models/test_actual.npy") / 1e5, np.load("models/test_pred.npy") / 1e5
fig, a = plt.subplots(figsize=(5.5, 5.5))
a.scatter(y, p, s=10, alpha=.5, color="#b3261e")
lim = [0, max(y.max(), p.max()) * 1.03]; a.plot(lim, lim, "k--", lw=1)
a.set_xlabel("Actual price (NPR Lakh)"); a.set_ylabel("Predicted price (NPR Lakh)")
a.set_title(f"Actual vs predicted – {best}"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("reports/fig_4_4_actual_vs_predicted.png", dpi=150)
print("Saved figures to reports/")
