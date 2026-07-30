"""Generate ablation study bar chart for GeoPredict component analysis."""
import matplotlib.pyplot as plt
import numpy as np

configs = [
    "Pi0 Baseline",
    "+ History Track\n  Encoder",
    "+ Future Track\n  Query",
    "+ Future Depth\n  (G_init only)",
    "+ Track + Depth\n  (no refine)",
    "+ Track-guided\n  Refinement\n  (Full Model)",
]
scores = [42.3, 44.8, 47.2, 49.4, 50.5, 52.4]
gains = [0, 2.5, 2.4, 2.2, 1.1, 1.9]
colors = ["#94a3b8", "#60a5fa", "#3b82f6", "#f97316", "#f59e0b", "#22c55e"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(configs)), scores, color=colors, edgecolor="white", linewidth=0.8)
for i, (bar, score, gain) in enumerate(zip(bars, scores, gains)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{score}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if gain > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2.5,
                f"+{gain}%", ha="center", va="top", fontsize=9, color="white", fontweight="bold")

ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=9, ha="center")
ax.set_ylabel("Average Success Rate (%)", fontsize=12)
ax.set_title("GeoPredict Component Ablation on RoboCasa", fontsize=13)
ax.set_ylim(35, 56)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("p/d/paper/asset/ablation_chart.png", dpi=150, bbox_inches="tight")
print("Saved ablation_chart.png")
