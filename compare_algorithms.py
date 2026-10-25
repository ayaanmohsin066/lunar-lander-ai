from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

TIMESTEPS = 1_000_000  # same budget for every algorithm — fair comparison
LOG_DIR = "./compare_tensorboard/"

algorithms = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

results = {}

for name, AlgoClass in algorithms.items():
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")

    env = make_vec_env("LunarLander-v3", n_envs=1)

    model = AlgoClass(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=LOG_DIR
    )

    model.learn(total_timesteps=TIMESTEPS, tb_log_name=name)
    model.save(f"{name.lower()}_lunarlander")

    eval_env = Monitor(gym.make("LunarLander-v3"))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    results[name] = (mean_reward, std_reward)
    print(f"{name} finished — mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")

print(f"\n{'='*50}\nFINAL COMPARISON\n{'='*50}")
for name, (mean_r, std_r) in results.items():
    print(f"{name:6s}: {mean_r:7.1f} +/- {std_r:.1f}")
