"""Generate comparison chart: VLA success rate improvement with 3D geometric priors.

Compares "without 3D" vs "with 3D" across papers and scenarios, sorted by
improvement magnitude. Data from published papers (2022-2026).
"""
import matplotlib.pyplot as plt
import numpy as np

# ---- Data (sorted by improvement, descending) ----
papers = [
    "SpatialVLA\n(Ego3D ablation)",
    "GeoPredict\n(real-geom.)",
    "SKIL\n(unseen obj.)",
    "FP3\n(RGB vs PC)",
    "3D Diffuser Actor\n(RLBench)",
    "GAM\n(cam. perturb.)",
    "ATM\n(130+ tasks)",
    "3DThinkVLA\n(real-height)",
    "GeoPredict\n(RoboCasa)",
    "PointACT\n(RLBench)",
    "QDepth-VLA\n(LIBERO)",
    "GeoPredict\n(LIBERO)",
    "G3VLA\n(LIBERO)",
]
without_3d = [37.5, 50.0, 30.0, 38.0, 47.0, 56.4, 37.0, 63.3, 42.3, 71.3, 88.8, 93.9, 95.0]
with_3d    = [87.5, 95.0, 72.8, 74.0, 81.3, 83.1, 63.0, 88.0, 52.4, 81.3, 96.5, 96.5, 97.0]
gains      = [w - wo for w, wo in zip(with_3d, without_3d)]

# ---- Colors by category ----
# A: training-only (green), B: explicit 3D input (blue), C: camera-aware (purple),
# D: scene policy (orange), E: keypoint constraint (teal), F: trajectory (pink)
category_colors = {
    "A": "#22c55e",  # training-only aux supervision
    "B": "#3b82f6",  # 3D point cloud input
    "C": "#8b5cf6",  # camera-aware encoding
    "D": "#f97316",  # 3D scene policy
    "E": "#14b8a6",  # keypoint constraint
    "F": "#ec4899",  # trajectory tracking
}
paper_categories = ["C", "A", "E", "B", "D", "C", "F", "A", "A", "B", "A", "A", "C"]
colors = [category_colors[c] for c in paper_categories]

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(papers))
width = 0.35

bars_wo = ax.bar(x - width / 2, without_3d, width, label="Without 3D",
                 color="#e2e8f0", edgecolor="#94a3b8", linewidth=0.8)
bars_w  = ax.bar(x + width / 2, with_3d, width, label="With 3D",
                 color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)

# Annotate gains
for i, (bar, gain) in enumerate(zip(bars_w, gains)):
    ax.annotate(
        f"+{gain:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
        color=colors[i],
    )

# Annotate baseline values
for bar, val in zip(bars_wo, without_3d):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color="#64748b")

ax.set_xticks(x)
ax.set_xticklabels(papers, fontsize=7.5, ha="center")
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title(
    "Impact of 3D Geometric Priors on VLA Success Rate\n"
    "(Sorted by Improvement Magnitude)",
    fontsize=13, fontweight="bold",
)
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=10)

# Category legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#22c55e", label="A: Training-Only Aux. Supervision"),
    Patch(facecolor="#3b82f6", label="B: Explicit 3D Input"),
    Patch(facecolor="#8b5cf6", label="C: Camera-Aware Encoding"),
    Patch(facecolor="#f97316", label="D: 3D Scene Policy"),
    Patch(facecolor="#14b8a6", label="E: Keypoint Constraint"),
    Patch(facecolor="#ec4899", label="F: Trajectory Tracking"),
]
ax2 = ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
                title="Category", title_fontsize=9, framealpha=0.9)
ax.add_artist(ax2)
ax.legend(["Without 3D", "With 3D"], loc="upper right", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("d:/SRC/Robot/GeoPredict/b/d/paper/asset/3d_keypoint_survey_chart.png",
            dpi=200, bbox_inches="tight")
print("Chart saved to asset/3d_keypoint_survey_chart.png")
plt.close()
