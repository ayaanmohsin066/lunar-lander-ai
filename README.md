# Lunar Lander RL — PPO, Reward Shaping, and a Statistically Validated Ablation

![Trained agent landing](landing.gif)

Training an agent to land a rocket in Gymnasium's `LunarLander-v3` using Stable-Baselines3, then stress-testing a reward-shaping idea with a 5-seed ablation and Welch's t-tests rather than trusting a single lucky run.

## Setup

- Python 3.11 (3.14 breaks Box2D wheels)
- `python3.11 -m venv lander-env && source lander-env/bin/activate`
- `pip install "gymnasium[box2d]" "stable-baselines3[extra]" torch scipy`

## Results

**Baseline PPO** (1M timesteps, 8 envs): 154.7 ± 122.6 reward over 20 eval episodes.

**Algorithm comparison** (1M timesteps, single env, 20 episodes):
| Algorithm | Reward |
|---|---|
| PPO | 165.8 ± 94.8 |
| DQN | 6.1 ± 72.0 |
| A2C | 42.5 ± 130.4 |

**Reward shaping** (penalizing angular velocity): a single-seed run looked like a clear win (258.1 ± 15.4, 0% crash rate vs. baseline's 17%). Touchdown metrics backed it up: final |angular velocity| dropped from 0.072 to 0.000, final |angle| from 0.114 to 0.002.

**Continuous control** (`LunarLanderContinuous-v3`, 1M timesteps, 100 episodes): 238.8 ± 56.9.

### Does the shaping result hold up? — 5-seed ablation

Ran each coefficient × 5 random seeds (25 training runs, 1M timesteps each) and compared against the unshaped baseline with Welch's t-test:

| Coefficient | Mean reward | Crash rate | p-value vs. baseline |
|---|---|---|---|
| 0.0 (baseline) | 230.9 ± 14.7 | 1.8% | — |
| 0.5 | 235.4 ± 9.9 | 1.2% | 0.6300 (not significant) |
| 2.0 | 223.8 ± 25.9 | 2.8% | 0.6509 (not significant) |
| 5.0 | 183.9 ± 4.9 | 8.8% | **0.0020 (significant)** |
| 10.0 | 162.4 ± 27.7 | 8.2% | **0.0046 (significant)** |

**Finding:** moderate shaping (coef ≤ 2.0) is statistically indistinguishable from baseline reward — the earlier single-seed "win" was a favorable outlier, not a real effect. Strong shaping (coef ≥ 5.0) significantly degrades reward and increases crash rate. This is the actual point of the exercise: a single seed can lie, and multi-seed validation with a significance test catches it.

## Files

- `train.py` — baseline PPO training
- `eval_baseline.py` — baseline evaluation
- `compare_algorithms.py` — PPO vs DQN vs A2C
- `reward_shaping.py` — angular-velocity-penalized reward wrapper
- `touchdown_metrics.py` — crash rate / final tilt / final angular velocity at touchdown
- `train_continuous.py` — continuous-action-space variant
- `reward_shaping_ablation_v2.py` — 5-seed × 5-coefficient ablation with Welch's t-tests

## Stack

Gymnasium, Stable-Baselines3 (PPO/DQN/A2C), PyTorch, SciPy, TensorBoard
