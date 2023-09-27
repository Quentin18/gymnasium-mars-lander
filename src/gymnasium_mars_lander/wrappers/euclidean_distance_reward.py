from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, Env, ObsType, WrapperActType, WrapperObsType

from gymnasium_mars_lander.envs import MarsLanderEnv
from gymnasium_mars_lander.envs.geometry import l2dist
from gymnasium_mars_lander.envs.rover_state import RoverState


class EuclideanDistanceRewardWrapper(gym.Wrapper):
    """Wrap the reward function to return the negative Euclidean distance between the
    rover and the middle point of the landing area at the end of the episode.
    """

    def __init__(self, env: Env[ObsType, ActType]) -> None:
        assert isinstance(
            env.unwrapped,
            MarsLanderEnv,
        ), "EuclideanDistanceWrapper must be used on MarsLanderEnv"
        super().__init__(env)

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        if not terminated and not truncated:
            return observation, reward, terminated, truncated, info

        env: MarsLanderEnv = self.env.unwrapped
        rover: RoverState = env.rover
        rover_pos = np.array([rover.x, rover.y])

        landing_area: np.ndarray = env.ground[env.landing_area]
        arrival_pos = np.mean(landing_area, axis=0)

        reward = -l2dist(rover_pos, arrival_pos)

        return observation, reward, terminated, truncated, info
