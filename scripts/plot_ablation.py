"""Figures from the results CSVs: ablation bars and the wind training curve."""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lander.evaluation import read_csv


def plot_ablation(summary_path, out):
    rows = read_csv(summary_path)
    coefs = [r["coef"] for r in rows]
    means = [float(r["mean_reward"]) for r in rows]
    sds = [float(r["mean_reward_sd_seeds"]) for r in rows]
    colors = ["#1f77b4"] + ["#d62728" if r["significant_bonferroni"] == "True" else "#2ca02c" for r in rows[1:]]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=sds, capsize=5, color=colors)
    ax.set_xticks(x, coefs)
    ax.set_xlabel("Reward shaping coefficient")
    ax.set_ylabel(f"Mean reward (± sd across {rows[0]['n_seeds']} seeds)")
    ax.set_title("Reward Shaping Ablation — LunarLander-v3 (PPO)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Baseline"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Not significant (Bonferroni)"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Significant (Bonferroni)"),
    ])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Saved", out)


def plot_wind(wind_path, out):
    by_seed = defaultdict(list)
    for r in read_csv(wind_path):
        by_seed[r["seed"]].append((int(r["timesteps"]) / 1e6, float(r["mean_reward"]),
                                   100 * (1 - float(r["landed_rate"]))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for seed, pts in sorted(by_seed.items()):
        pts.sort()
        t, rew, fail = zip(*pts)
        ax1.plot(t, rew, marker="o", label=f"seed {seed}")
        ax2.plot(t, fail, marker="o", label=f"seed {seed}")
    ax1.set(xlabel="Training timesteps (M)", ylabel="Mean reward, wind=15 / turb=1.5", title="Windy reward")
    ax2.set(xlabel="Training timesteps (M)", ylabel="% episodes not landed", title="Failure rate")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Saved", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation", default="results/ablation_summary.csv")
    p.add_argument("--wind", default="results/wind.csv")
    p.add_argument("--fig-dir", default="figures")
    a = p.parse_args()
    if os.path.exists(a.ablation):
        plot_ablation(a.ablation, f"{a.fig_dir}/ablation_results.png")
    if os.path.exists(a.wind):
        plot_wind(a.wind, f"{a.fig_dir}/wind_robust_curve.png")


if __name__ == "__main__":
    main()
