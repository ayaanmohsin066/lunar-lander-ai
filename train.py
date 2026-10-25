from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v3", n_envs=8)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./lander_tensorboard/"
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_lunarlander")
