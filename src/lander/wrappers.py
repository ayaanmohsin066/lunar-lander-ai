import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """Subtracts coefficient * |angular velocity| from every step's reward.

    Note: the environment's own shaping already contains -100 * |angle|, so this
    penalises the time derivative of a quantity the default reward already penalises.
    """

    def __init__(self, env, coefficient):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= abs(obs[5]) * self.coefficient  # obs[5] = angular velocity
        return obs, reward, terminated, truncated, info


class RandomWindWrapper(gym.Wrapper):
    """Domain randomisation: draws new wind and turbulence strengths every episode.

    Draws come from the env's own seeded generator (self.np_random), taken *after*
    the inner reset so that reset(seed=...) has already reseeded it. wind_power and
    turbulence_power are only read in step(), so setting them after reset is safe.
    """

    def __init__(self, env, wind_power_range=(0.0, 20.0), turbulence_range=(0.0, 2.0)):
        super().__init__(env)
        self.wind_power_range = wind_power_range
        self.turbulence_range = turbulence_range

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.unwrapped.wind_power = self.np_random.uniform(*self.wind_power_range)
        self.unwrapped.turbulence_power = self.np_random.uniform(*self.turbulence_range)
        return obs, info


def make_shaped_env(coefficient, env_id="LunarLander-v3"):
    return CustomRewardWrapper(gym.make(env_id), coefficient)


WINDY_EVAL_KWARGS = dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5)


def make_wind_randomised_env():
    return RandomWindWrapper(gym.make("LunarLander-v3", **WINDY_EVAL_KWARGS))
