import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from scipy import stats
import numpy as np

COEFFICIENTS = [0.0, 0.5, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 1_000_000
N_EVAL_EPISODES = 100

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, coefficient):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angular_velocity = obs[5]
        reward -= abs(angular_velocity) * self.coefficient
        return obs, reward, terminated, truncated, info


def make_env(coefficient):
    def _init():
        return CustomRewardWrapper(gym.make("LunarLander-v3"), coefficient)
    return _init


def evaluate_touchdown(model, n_episodes=N_EVAL_EPISODES):
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

        final_angles.append(abs(obs[4]))
        final_angular_velocities.append(abs(obs[5]))
        total_rewards.append(episode_reward)
        if episode_reward < 0:
            crashes += 1

    env.close()
    return {
        "crash_rate": crashes / n_episodes,
        "mean_reward": np.mean(total_rewards),
        "mean_final_angular_velocity": np.mean(final_angular_velocities),
        "mean_final_angle": np.mean(final_angles),
    }


all_results = {coef: [] for coef in COEFFICIENTS}

for coef in COEFFICIENTS:
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"coef={coef}, seed={seed}")
        print(f"{'='*60}")

        env = make_vec_env(make_env(coef), n_envs=8, seed=seed)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                     tensorboard_log="./ablation_v2_tensorboard/")
        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"PPO_coef_{coef}_seed_{seed}")

        metrics = evaluate_touchdown(model)
        all_results[coef].append(metrics)
        print(f"  reward={metrics['mean_reward']:.1f}, crash_rate={metrics['crash_rate']*100:.1f}%, "
              f"final_ang_vel={metrics['mean_final_angular_velocity']:.3f}")

print(f"\n{'='*80}")
print("ABLATION SUMMARY (mean +/- std across 5 seeds)")
print(f"{'='*80}")
print(f"{'Coef':<8}{'Reward':<20}{'Crash %':<18}{'Final |w|':<12}")

reward_distributions = {}
for coef in COEFFICIENTS:
    runs = all_results[coef]
    rewards = [r['mean_reward'] for r in runs]
    crashes = [r['crash_rate']*100 for r in runs]
    ang_vels = [r['mean_final_angular_velocity'] for r in runs]
    reward_distributions[coef] = rewards
    print(f"{coef:<8}{np.mean(rewards):.1f}+/-{np.std(rewards):.1f}".ljust(28)
          + f"{np.mean(crashes):.1f}+/-{np.std(crashes):.1f}".ljust(18)
          + f"{np.mean(ang_vels):.3f}")

print(f"\n{'='*80}")
print("STATISTICAL SIGNIFICANCE: coef=0.0 (control) vs each other coefficient")
print(f"Welch's t-test on reward distributions across 5 seeds each")
print(f"{'='*80}")
baseline_rewards = reward_distributions[0.0]
for coef in COEFFICIENTS[1:]:
    t_stat, p_value = stats.ttest_ind(baseline_rewards, reward_distributions[coef], equal_var=False)
    sig = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "not significant"
    print(f"coef=0.0 vs coef={coef}: t={t_stat:.2f}, p={p_value:.4f} -> {sig}")
