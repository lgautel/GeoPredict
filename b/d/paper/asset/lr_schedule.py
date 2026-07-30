"""Generate LR schedule curve for GeoPredict training."""
import math
import matplotlib.pyplot as plt
import numpy as np

base_lr = 2.5e-5
warmup_steps = 1000
decay_steps = 30000
decay_lr = 2.5e-6
total_steps = 40000
init_lr = base_lr / (warmup_steps + 1)

steps = np.arange(total_steps)
lrs = []
for step in steps:
    if step < warmup_steps:
        lr = init_lr + (base_lr - init_lr) * step / warmup_steps
    elif step < decay_steps:
        cosine_steps = decay_steps - warmup_steps
        progress = (step - warmup_steps) / cosine_steps
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        lr = decay_lr + (base_lr - decay_lr) * cosine_factor
    else:
        lr = decay_lr
    lrs.append(lr)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, lrs, color="#2563eb", linewidth=1.5)
ax.axvline(x=warmup_steps, color="#dc2626", linestyle="--", alpha=0.6, label=f"Warmup end ({warmup_steps})")
ax.axvline(x=decay_steps, color="#16a34a", linestyle="--", alpha=0.6, label=f"Decay end ({decay_steps})")
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("GeoPredict LR Schedule (Linear Warmup + Cosine Decay)", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, total_steps)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("p/d/paper/asset/lr_schedule.png", dpi=150, bbox_inches="tight")
print("Saved lr_schedule.png")
