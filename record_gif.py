import gymnasium as gym
from stable_baselines3 import PPO
import imageio

MODEL_PATH = "ppo_lunarlander_shaped"  # change to whichever model you want to show off
OUTPUT = "landing.gif"

env = gym.make("LunarLander-v3", render_mode="rgb_array")
model = PPO.load(MODEL_PATH)

obs, _ = env.reset(seed=42)
frames = []
done = False
while not done:
    frames.append(env.render())
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
imageio.mimsave(OUTPUT, frames, fps=30)
print(f"Saved {len(frames)} frames to {OUTPUT}")
