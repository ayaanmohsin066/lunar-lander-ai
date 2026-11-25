"""Record one deterministic episode of a saved policy to a GIF."""
import argparse

import gymnasium as gym
import imageio
from stable_baselines3 import PPO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_lunarlander_shaped")
    p.add_argument("--out", default="figures/landing.gif")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = PPO.load(a.model)
    obs, _ = env.reset(seed=a.seed)
    frames, done = [], False
    while not done:
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave(a.out, frames, fps=30)
    print(f"Saved {len(frames)} frames to {a.out}")


if __name__ == "__main__":
    main()
