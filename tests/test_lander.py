import gymnasium as gym
import numpy as np
import pytest

from lander.evaluation import classify_outcome, summarize
from lander.stats import cohens_d
from lander.wrappers import RandomWindWrapper, WINDY_EVAL_KWARGS


def _rollout(policy, seed, max_steps=1000):
    env = gym.make("LunarLander-v3", max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
    return classify_outcome(env, obs, terminated), reward, truncated


POLICIES = {
    "noop": lambda o: 0,
    "side_engine": lambda o: 1,
    "main_engine": lambda o: 2,
    # crude hover: fire main engine when falling
    "hover": lambda o: 2 if o[3] < -0.05 else 0,
}


def test_classifier_agrees_with_env_terminal_reward():
    seen = set()
    for name, pol in POLICIES.items():
        for seed in range(15):
            outcome, last_reward, truncated = _rollout(pol, seed, max_steps=300)
            seen.add(outcome)
            if outcome == "timeout":
                assert truncated
            elif outcome == "landed":
                assert last_reward == 100
            else:  # crash / out_of_bounds
                assert last_reward == -100
    # the cases the old reward<0 rule conflated must actually occur
    assert {"crash", "out_of_bounds", "timeout"} <= seen, seen


def test_cohens_d_uses_sample_std():
    a, b = [1.0, 2, 3, 4, 5], [3.0, 4, 5, 6, 7]
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    assert cohens_d(a, b) == pytest.approx(-2 / pooled)
    # ddof=0 would overstate |d| by sqrt(n/(n-1))
    ddof0 = -2 / np.sqrt((np.var(a) + np.var(b)) / 2)
    assert abs(ddof0 / cohens_d(a, b)) == pytest.approx(np.sqrt(5 / 4))


def test_wind_is_reproducible_from_reset_seed_and_ignores_global_rng():
    def draws(seed):
        env = RandomWindWrapper(gym.make("LunarLander-v3", **WINDY_EVAL_KWARGS))
        out = []
        env.reset(seed=seed)
        out.append((env.unwrapped.wind_power, env.unwrapped.turbulence_power))
        np.random.seed(123)  # perturb global RNG between episodes
        env.reset()
        out.append((env.unwrapped.wind_power, env.unwrapped.turbulence_power))
        return out
    np.random.seed(1)
    first = draws(7)
    np.random.seed(999)
    assert draws(7) == first
    assert draws(8) != first
    assert all(0 <= w <= 20 and 0 <= t <= 2 for w, t in first)


def test_summarize_conditions_on_landed():
    eps = [
        dict(reward=200, outcome="landed", legacy_crash=False, contact_abs_angle=0.1,
             contact_abs_ang_vel=0.2, final_abs_angle=0.0, final_abs_ang_vel=0.0),
        dict(reward=-50, outcome="timeout", legacy_crash=True, contact_abs_angle=float("nan"),
             contact_abs_ang_vel=float("nan"), final_abs_angle=0.3, final_abs_ang_vel=0.5),
    ]
    s = summarize(eps)
    assert s["landed_rate"] == 0.5 and s["timeout_rate"] == 0.5 and s["crash_rate"] == 0
    assert s["legacy_crash_rate"] == 0.5
    assert s["landed_contact_abs_ang_vel"] == 0.2
    assert s["legacy_final_abs_ang_vel"] == 0.25
