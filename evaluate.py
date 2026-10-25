from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("ppo_lunarlander")
env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
