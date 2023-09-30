from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType


class NegativeRewardWrapper(gym.Wrapper):
    """Gives a reward of -1 at each step to urge the agent to reach a terminal state
    as soon as possible to avoid penalties."""

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward = -1 if not terminated and not truncated else reward

        return observation, reward, terminated, truncated, info
