from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

TIMESTEPS = 1_000_000
LOG_DIR = "./continuous_tensorboard/"

env = make_vec_env("LunarLanderContinuous-v3", n_envs=8)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR
)

model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_continuous")
model.save("ppo_lunarlander_continuous")

eval_env = Monitor(gym.make("LunarLanderContinuous-v3"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"\nContinuous PPO — evaluated: {mean_reward:.1f} +/- {std_reward:.1f}")
