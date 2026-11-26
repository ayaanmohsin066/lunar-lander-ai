"""Episode evaluation with an outcome classifier that reads the simulator state.

The old code called any episode with total reward < 0 a "crash". That also counts
timeouts (no terminal bonus, only fuel cost) and lateral flyaways, and can miss or
include off-pad landings. LunarLander terminates for exactly three reasons
(game_over, |x| >= 1, lander asleep); the 1000-step limit is TimeLimit truncation.
"""
import csv
import math
import os

import gymnasium as gym
import numpy as np

OUTCOMES = ("landed", "crash", "out_of_bounds", "timeout")


def classify_outcome(env, obs, terminated):
    if not terminated:
        return "timeout"
    u = env.unwrapped
    # Same tests the env uses to assign its terminal reward. If game_over and asleep
    # coincide the env pays +100; we still call body-ground contact a crash.
    if u.game_over:
        return "crash"
    if abs(obs[0]) >= 1.0:
        return "out_of_bounds"
    if not u.lander.awake:
        return "landed"
    raise RuntimeError("terminated for an unknown reason")


def run_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    total, contact = 0.0, None
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if contact is None and (obs[6] or obs[7]):  # first leg-ground contact
            contact = (abs(float(obs[4])), abs(float(obs[5])))
    return {
        "seed": seed,
        "reward": total,
        "outcome": classify_outcome(env, obs, terminated),
        "legacy_crash": total < 0,  # the old classifier, kept for comparison
        "contact_abs_angle": contact[0] if contact else math.nan,
        "contact_abs_ang_vel": contact[1] if contact else math.nan,
        "final_abs_angle": abs(float(obs[4])),
        "final_abs_ang_vel": abs(float(obs[5])),
    }


def evaluate(model, n_episodes=100, env_id="LunarLander-v3", seed_offset=0, **env_kwargs):
    """Runs n deterministic episodes; episode i is reset with seed seed_offset + i."""
    env = gym.make(env_id, **env_kwargs)
    try:
        return [run_episode(model, env, seed_offset + i) for i in range(n_episodes)]
    finally:
        env.close()


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else math.nan


def summarize(episodes):
    rewards = [e["reward"] for e in episodes]
    landed = [e for e in episodes if e["outcome"] == "landed"]
    out = {
        "n_episodes": len(episodes),
        "mean_reward": float(np.mean(rewards)),
        # across-episode spread for this one model (sample std)
        "std_reward_episodes": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else math.nan,
    }
    for o in OUTCOMES:
        out[f"{o}_rate"] = sum(e["outcome"] == o for e in episodes) / len(episodes)
    out["legacy_crash_rate"] = sum(e["legacy_crash"] for e in episodes) / len(episodes)
    out["n_with_contact"] = sum(not math.isnan(e["contact_abs_angle"]) for e in episodes)
    out["contact_abs_angle"] = _mean([e["contact_abs_angle"] for e in episodes])
    out["contact_abs_ang_vel"] = _mean([e["contact_abs_ang_vel"] for e in episodes])
    out["landed_contact_abs_angle"] = _mean([e["contact_abs_angle"] for e in landed])
    out["landed_contact_abs_ang_vel"] = _mean([e["contact_abs_ang_vel"] for e in landed])
    out["landed_final_abs_angle"] = _mean([e["final_abs_angle"] for e in landed])
    out["landed_final_abs_ang_vel"] = _mean([e["final_abs_ang_vel"] for e in landed])
    # the old, unconditioned metric (all episodes, at termination)
    out["legacy_final_abs_angle"] = _mean([e["final_abs_angle"] for e in episodes])
    out["legacy_final_abs_ang_vel"] = _mean([e["final_abs_ang_vel"] for e in episodes])
    return out


def write_csv(path, rows, append=False):
    """Writes a list of dicts. With append=True, adds to an existing file (header once)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = append and os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a" if append else "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))  # union, in order
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def record(episodes, config, summary_path, episodes_path=None):
    """Appends one summary row (config + summarize) and, optionally, every episode."""
    summary = {**config, **summarize(episodes)}
    write_csv(summary_path, [summary], append=True)
    if episodes_path:
        write_csv(episodes_path, [{**config, **e} for e in episodes], append=True)
    return summary


def describe(s):
    return (f"reward={s['mean_reward']:.1f} (sd over episodes {s['std_reward_episodes']:.1f}), "
            f"landed={s['landed_rate']:.0%} crash={s['crash_rate']:.0%} "
            f"oob={s['out_of_bounds_rate']:.0%} timeout={s['timeout_rate']:.0%} "
            f"(legacy reward<0 rate={s['legacy_crash_rate']:.0%})")
