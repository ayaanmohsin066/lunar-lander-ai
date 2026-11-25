"""Reward-shaping ablation: every coefficient x every seed, 1M PPO steps each.

Writes one row per (coef, seed) to results/ablation_runs.csv and every evaluation
episode to results/ablation_episodes.csv. Statistics live in scripts/effect_size.py.
"""
import argparse
import itertools
from multiprocessing import Pool

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from lander.evaluation import describe, evaluate, record
from lander.wrappers import make_shaped_env

def run(job):
    import torch
    torch.set_num_threads(1)  # one core per worker
    A, coef, seed = job
    env = make_vec_env(lambda: make_shaped_env(coef), n_envs=A.n_envs, seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu", tensorboard_log=A.tb_log)
    model.learn(total_timesteps=A.timesteps, tb_log_name=f"PPO_coef_{coef}_seed_{seed}")
    if A.model_dir:
        model.save(f"{A.model_dir}/ablation_coef{coef}_seed{seed}")
    return {"coef": coef, "seed": seed, "timesteps": A.timesteps, "n_envs": A.n_envs}, evaluate(model, A.n_eval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coefs", type=float, nargs="+", default=[0.0, 0.5, 2.0, 5.0, 10.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--workers", type=int, default=1, help="training runs in parallel")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results/ablation_runs.csv")
    p.add_argument("--episodes-out", default="results/ablation_episodes.csv")
    p.add_argument("--tb-log", default=None)
    A = p.parse_args()

    jobs = [(A, c, s) for c, s in itertools.product(A.coefs, A.seeds)]
    with Pool(A.workers) as pool:
        # rows are written as each run finishes, so a crash loses at most the runs in flight
        for config, episodes in pool.imap_unordered(run, jobs):
            s = record(episodes, config, A.out, A.episodes_out)
            print(f"coef={config['coef']} seed={config['seed']}:", describe(s), flush=True)


if __name__ == "__main__":
    main()
