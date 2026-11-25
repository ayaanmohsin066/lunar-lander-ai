"""Sanity check that Box2D and rendering work: a random policy in a window."""
import argparse

import gymnasium as gym


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args()
    env = gym.make("LunarLander-v3", render_mode="human")
    env.reset()
    for _ in range(a.steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
