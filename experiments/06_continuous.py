"""PPO on the continuous-action variant, LunarLanderContinuous-v3."""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from lander.evaluation import describe, evaluate, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--model", default="models/ppo_lunarlander_continuous")
    p.add_argument("--out", default="results/continuous.csv")
    p.add_argument("--tb-log", default=None)
    a = p.parse_args()

    env = make_vec_env("LunarLanderContinuous-v3", n_envs=a.n_envs, seed=a.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=a.seed, tensorboard_log=a.tb_log)
    model.learn(total_timesteps=a.timesteps)
    model.save(a.model)
    config = {"experiment": "continuous", "seed": a.seed, "timesteps": a.timesteps, "n_envs": a.n_envs}
    s = record(evaluate(model, a.n_eval, env_id="LunarLanderContinuous-v3"), config, a.out)
    print("Continuous PPO:", describe(s))


if __name__ == "__main__":
    main()
