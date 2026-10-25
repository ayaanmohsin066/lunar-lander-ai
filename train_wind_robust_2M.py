import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RandomWindWrapper(gym.Wrapper):
    def __init__(self, env, wind_power_range=(0.0, 20.0), turbulence_range=(0.0, 2.0)):
        super().__init__(env)
        self.wind_power_range = wind_power_range
        self.turbulence_range = turbulence_range

    def reset(self, **kwargs):
        self.unwrapped.wind_power = np.random.uniform(*self.wind_power_range)
        self.unwrapped.turbulence_power = np.random.uniform(*self.turbulence_range)
        return self.env.reset(**kwargs)

def make_env():
    env = gym.make("LunarLander-v3", enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    return RandomWindWrapper(env)

def evaluate_windy(model, n_episodes=100):
    env = gym.make("LunarLander-v3", enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    rewards, crashes = [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, total_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        if total_reward < 0:
            crashes += 1
    env.close()
    return np.mean(rewards), np.std(rewards), (crashes / n_episodes) * 100

env = make_vec_env(make_env, n_envs=8, seed=0)
model = PPO("MlpPolicy", env, verbose=0, seed=0)
model.learn(total_timesteps=2_000_000)
mean_r, std_r, crash_pct = evaluate_windy(model)
print(f"2M timesteps: windy_reward={mean_r:.1f} +/- {std_r:.1f}, crash_rate={crash_pct:.1f}%")
