from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

model = PPO.load("ppo_lunarlander")
eval_env = Monitor(gym.make("LunarLander-v3"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"Baseline PPO (unshaped) — evaluated on original reward: {mean_reward:.1f} +/- {std_reward:.1f}")
