"""PPO vs DQN vs A2C, same timestep budget, single training env each."""
import argparse

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from lander.evaluation import describe, evaluate, record

ALGOS = {"PPO": PPO, "DQN": DQN, "A2C": A2C}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results/algorithms.csv")
    p.add_argument("--tb-log", default=None)
    a = p.parse_args()

    for name in a.algos:
        env = make_vec_env("LunarLander-v3", n_envs=1, seed=a.seed)
        model = ALGOS[name]("MlpPolicy", env, verbose=0, seed=a.seed, tensorboard_log=a.tb_log)
        model.learn(total_timesteps=a.timesteps, tb_log_name=name)
        model.save(f"{a.model_dir}/{name.lower()}_lunarlander")
        config = {"algorithm": name, "seed": a.seed, "timesteps": a.timesteps, "n_envs": 1}
        s = record(evaluate(model, a.n_eval), config, a.out)
        print(f"{name}:", describe(s))


if __name__ == "__main__":
    main()
