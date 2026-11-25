"""Wind-robust PPO: train on randomised wind, evaluate at wind_power=15, turbulence=1.5.

Trains in chunks and evaluates after each, so one run gives 1M, 2M, ... checkpoints.
  3-seed 1M/2M table:  --seeds 0 1 2 --n-chunks 2
  long-run curve:      --seeds 0 --n-chunks 10
Training randomises wind_power in [0, 20] and turbulence in [0, 2]; evaluation uses a
single point of that range, so results describe that condition, not wind generally.
"""
import argparse
from multiprocessing import Pool

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from lander.evaluation import describe, evaluate, record
from lander.wrappers import WINDY_EVAL_KWARGS, make_wind_randomised_env

def run(job):
    A, seed = job
    import torch
    torch.set_num_threads(1)
    env = make_vec_env(make_wind_randomised_env, n_envs=A.n_envs, seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu")
    rows = []
    for i in range(1, A.n_chunks + 1):
        model.learn(total_timesteps=A.chunk, reset_num_timesteps=False)
        config = {"seed": seed, "timesteps": i * A.chunk, "n_envs": A.n_envs}
        rows.append((config, evaluate(model, A.n_eval, **WINDY_EVAL_KWARGS)))
    if A.model_dir:
        model.save(f"{A.model_dir}/ppo_wind_robust_seed{seed}_{A.n_chunks * A.chunk}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chunk", type=int, default=1_000_000, help="timesteps between evaluations")
    p.add_argument("--n-chunks", type=int, default=2)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results/wind.csv")
    p.add_argument("--episodes-out", default="results/wind_episodes.csv")
    A = p.parse_args()

    with Pool(A.workers) as pool:
        for rows in pool.imap_unordered(run, [(A, s) for s in A.seeds]):
            for config, episodes in rows:
                s = record(episodes, config, A.out, A.episodes_out)
                print(f"seed={config['seed']} {config['timesteps']:,} steps:", describe(s), flush=True)


if __name__ == "__main__":
    main()
