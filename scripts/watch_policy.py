"""Watch a saved policy fly in a window."""
import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_lunarlander")
    p.add_argument("--steps", type=int, default=1000)
    a = p.parse_args()

    model = PPO.load(a.model)
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = env.reset()
    for _ in range(a.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
