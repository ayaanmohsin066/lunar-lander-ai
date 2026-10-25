import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class CustomRewardWrapper(gym.Wrapper):
    """Adds an extra penalty for high angular velocity, pushing the
    agent toward gentler, more stable landings instead of just
    optimizing for the default reward."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angular_velocity = obs[5]  # LunarLander obs index 5 = angular velocity
        reward -= abs(angular_velocity) * 2.0
        return obs, reward, terminated, truncated, info

def make_env():
    return CustomRewardWrapper(gym.make("LunarLander-v3"))

TIMESTEPS = 1_000_000
LOG_DIR = "./reward_shaping_tensorboard/"

env = make_vec_env(make_env, n_envs=8)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR
)

model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_shaped")
model.save("ppo_lunarlander_shaped")

eval_env = Monitor(gym.make("LunarLander-v3"))
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"\nShaped-reward PPO — evaluated on original reward: {mean_reward:.1f} +/- {std_reward:.1f}")
