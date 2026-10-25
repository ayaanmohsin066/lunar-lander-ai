import matplotlib.pyplot as plt
import numpy as np

coefficients = [0.0, 0.5, 2.0, 5.0, 10.0]
means = [230.9, 235.4, 223.8, 183.9, 162.4]
stds = [14.7, 9.9, 25.9, 4.9, 27.7]
significant = [False, False, False, True, True]  # vs. baseline, p<0.05

colors = ['#d62728' if sig else '#2ca02c' for sig in significant]
colors[0] = '#1f77b4'  # baseline in its own color

x = np.arange(len(coefficients))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, means, yerr=stds, capsize=5, color=colors)
ax.set_xticks(x)
ax.set_xticklabels([str(c) for c in coefficients])
ax.set_xlabel("Reward shaping coefficient")
ax.set_ylabel("Mean reward (5 seeds, 100 eval episodes)")
ax.set_title("Reward Shaping Ablation — LunarLander-v3 (PPO)")
ax.axhline(means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline mean')
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color='#1f77b4', label='Baseline'),
    plt.Rectangle((0,0),1,1, color='#2ca02c', label='Not significant (p≥0.05)'),
    plt.Rectangle((0,0),1,1, color='#d62728', label='Significant (p<0.05)'),
])
plt.tight_layout()
plt.savefig("ablation_results.png", dpi=150)
print("Saved ablation_results.png")
