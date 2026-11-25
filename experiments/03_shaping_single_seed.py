"""Single-seed shaped PPO (angular-velocity penalty), compared against a saved baseline.

Superseded as evidence by 04_shaping_ablation.py (5 seeds per arm); kept because the
README's original "looks like a win" result came from this setup.
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from lander.evaluation import describe, evaluate, record
from lander.wrappers import make_shaped_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coef", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--baseline-model", default="models/ppo_lunarlander",
                   help="model from 01_baseline.py to compare against ('' to skip)")
    p.add_argument("--model", default="models/ppo_lunarlander_shaped")
    p.add_argument("--out", default="results/shaping_single_seed.csv")
    p.add_argument("--tb-log", default=None)
    a = p.parse_args()

    env = make_vec_env(lambda: make_shaped_env(a.coef), n_envs=a.n_envs, seed=a.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=a.seed, tensorboard_log=a.tb_log)
    model.learn(total_timesteps=a.timesteps)
    model.save(a.model)

    models = {"shaped": model}
    if a.baseline_model:
        models["baseline"] = PPO.load(a.baseline_model)
    for name, m in models.items():
        config = {"model": name, "coef": a.coef if name == "shaped" else 0.0, "seed": a.seed}
        s = record(evaluate(m, a.n_eval), config, a.out)  # evaluated on the original reward
        print(f"{name}:", describe(s))
        print(f"  |ang vel| at first contact, landed only: {s['landed_contact_abs_ang_vel']:.3f}; "
              f"legacy final |ang vel| (all episodes): {s['legacy_final_abs_ang_vel']:.3f}")


if __name__ == "__main__":
    main()
