"""Ablation statistics from results/ablation_runs.csv.

Per coefficient: mean and sample std (ddof=1) of the per-seed mean rewards, outcome
rates, Welch's t-test and Cohen's d vs the coef=0 baseline, and a Bonferroni check
for the number of comparisons. Writes results/ablation_summary.csv.
"""
import argparse
from collections import defaultdict

import numpy as np

from lander.evaluation import write_csv, read_csv
from lander.stats import cohens_d, magnitude, welch_p

PER_SEED = ["mean_reward", "landed_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
            "legacy_crash_rate", "landed_contact_abs_angle", "landed_contact_abs_ang_vel",
            "landed_final_abs_ang_vel", "legacy_final_abs_angle", "legacy_final_abs_ang_vel"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/ablation_runs.csv")
    p.add_argument("--out", default="results/ablation_summary.csv")
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()

    by_coef = defaultdict(list)
    for r in read_csv(a.runs):
        by_coef[float(r["coef"])].append(r)
    coefs = sorted(by_coef)
    base = [float(r["mean_reward"]) for r in by_coef[0.0]]
    alpha_bonf = a.alpha / (len(coefs) - 1)

    rows = []
    for c in coefs:
        runs = by_coef[c]
        rewards = [float(r["mean_reward"]) for r in runs]
        row = {"coef": c, "n_seeds": len(runs)}
        for k in PER_SEED:
            v = np.array([float(r[k]) for r in runs])
            row[k] = np.nanmean(v)
            row[k + "_sd_seeds"] = np.nanstd(v, ddof=1)  # across-seed sample std
        if c != 0.0:
            row["welch_p"] = welch_p(base, rewards)
            row["cohens_d"] = cohens_d(rewards, base)
            row["significant"] = row["welch_p"] < a.alpha
            row["significant_bonferroni"] = row["welch_p"] < alpha_bonf
        rows.append(row)
    write_csv(a.out, rows)

    print(f"Bonferroni alpha for {len(coefs) - 1} comparisons: {alpha_bonf:.4f}")
    print(f"{'coef':<6}{'reward (mean ± sd across seeds)':<34}{'landed':<8}{'crash':<8}{'oob':<7}"
          f"{'timeout':<9}{'legacy':<8}{'p':<9}{'d':<14}")
    for r in rows:
        stats = ""
        if "welch_p" in r:
            stats = f"{r['welch_p']:<9.4f}{r['cohens_d']:+.2f} {magnitude(r['cohens_d'])}"
            stats += " *" if r["significant_bonferroni"] else ""
        print(f"{r['coef']:<6}{r['mean_reward']:.1f} ± {r['mean_reward_sd_seeds']:.1f}".ljust(40)
              + f"{r['landed_rate']:<8.1%}{r['crash_rate']:<8.1%}{r['out_of_bounds_rate']:<7.1%}"
              + f"{r['timeout_rate']:<9.1%}{r['legacy_crash_rate']:<8.1%}" + stats)
    print("* = significant after Bonferroni correction")
    print(f"\n{'coef':<6}{'|w| at contact (landed)':<26}{'|angle| at contact (landed)':<30}{'legacy final |w|':<18}")
    for r in rows:
        print(f"{r['coef']:<6}{r['landed_contact_abs_ang_vel']:<26.3f}{r['landed_contact_abs_angle']:<30.3f}"
              f"{r['legacy_final_abs_ang_vel']:<18.3f}")


if __name__ == "__main__":
    main()
