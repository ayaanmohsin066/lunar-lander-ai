import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

def evaluate_touchdown(model_path, n_episodes=100):
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")

    crashes = 0
    final_angular_velocities = []
    final_angles = []
    total_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        # obs at the final step: index 4 = angle, index 5 = angular velocity
        final_angles.append(abs(obs[4]))
        final_angular_velocities.append(abs(obs[5]))
        total_rewards.append(episode_reward)

        # A crash ends the episode with the environment's -100 penalty baked in;
        # a safe landing ends with +100. Episode reward reliably separates them.
        if episode_reward < 0:
            crashes += 1

    env.close()

    return {
        "crash_rate": crashes / n_episodes,
        "mean_reward": np.mean(total_rewards),
        "mean_final_angular_velocity": np.mean(final_angular_velocities),
        "mean_final_angle": np.mean(final_angles),
    }

print("Evaluating baseline PPO...")
baseline = evaluate_touchdown("ppo_lunarlander")

print("Evaluating shaped PPO...")
shaped = evaluate_touchdown("ppo_lunarlander_shaped")

print(f"\n{'='*60}")
print(f"{'Metric':<30}{'Baseline':<15}{'Shaped':<15}")
print(f"{'='*60}")
print(f"{'Crash rate':<30}{baseline['crash_rate']*100:<15.1f}{shaped['crash_rate']*100:<15.1f}")
print(f"{'Mean reward':<30}{baseline['mean_reward']:<15.1f}{shaped['mean_reward']:<15.1f}")
print(f"{'Mean |angular velocity| @ end':<30}{baseline['mean_final_angular_velocity']:<15.3f}{shaped['mean_final_angular_velocity']:<15.3f}")
print(f"{'Mean |angle| @ end':<30}{baseline['mean_final_angle']:<15.3f}{shaped['mean_final_angle']:<15.3f}")
