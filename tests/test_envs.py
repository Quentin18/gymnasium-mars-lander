import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gymnasium_mars_lander.envs import MarsLanderEnv
from gymnasium_mars_lander.envs.rover_state import RoverState


@pytest.fixture
def env() -> MarsLanderEnv:
    return MarsLanderEnv()


def test_check_env():
    env = gym.make("gymnasium_mars_lander/MarsLander-v0")
    check_env(env=env.unwrapped)


def test_reset(env: MarsLanderEnv):
    # Given
    ground = np.array(
        [
            [0, 100],
            [1000, 500],
            [1500, 100],
            [3000, 100],
            [3500, 100],
            [3700, 200],
            [5000, 1500],
            [5800, 100],
            [6000, 120],
            [6999, 2000],
        ]
    )
    rover = np.random.rand(7)

    # When
    env.reset(options={"ground": ground, "rover": rover})

    # Then
    assert hasattr(env, "ground")
    assert env.ground.shape == (30, 2)
    assert all(point in env.ground for point in ground)
    assert hasattr(env, "rover")
    assert hasattr(env, "landing_area")
    assert env.landing_area == [6, 11]


def test_update_state(env: MarsLanderEnv):
    # Given
    env.rover = RoverState(state=(6500, 2800, -100, 0, 600, 90, 0))

    # When
    for i in range(3):
        env._update_state(rotate=-15, power=1)
    for i in range(4):
        env._update_state(rotate=-15, power=0)
    env._update_state(rotate=-5, power=0)
    for i in range(60):
        env._update_state(rotate=0, power=0)
    state = np.rint(env.rover.numpy())

    # Then
    np.testing.assert_array_equal(state, [1191, 405, -44, -66, 399, -20, 3])


def test_collision(env: MarsLanderEnv):
    # Given
    # https://www.codingame.com/ide/puzzle/mars-lander - Initial speed, correct side
    env.reset(
        options={
            "ground": np.array(
                [
                    [0, 100],
                    [1000, 500],
                    [1500, 100],
                    [3000, 100],
                    [3500, 100],
                    [3700, 200],
                    [5000, 1500],
                    [5800, 100],
                    [6000, 120],
                    [6999, 2000],
                ]
            ),
            "rover": (6500, 2800, -100, 0, 600, 90, 0),
        }
    )

    # When
    for i in range(70):
        observation, reward, terminated, truncated, info = env.step([-1, 1])
        if terminated or truncated:
            break

    # Then
    assert i == 39
    assert reward < 0


def test_rover_exits_field_of_view(env: MarsLanderEnv):
    # Given
    # https://www.codingame.com/ide/puzzle/mars-lander - Initial speed, correct side
    env.reset(
        options={
            "ground": np.array(
                [
                    [0, 100],
                    [1000, 500],
                    [1500, 100],
                    [3000, 100],
                    [3500, 100],
                    [3700, 200],
                    [5000, 1500],
                    [5800, 100],
                    [6000, 120],
                    [6999, 2000],
                ]
            ),
            "rover": (6500, 2800, -100, 0, 600, 90, 0),
        }
    )

    # When
    for i in range(6):
        observation, reward, terminated, truncated, info = env.step([-1, 1])
    for i in range(64):
        observation, reward, terminated, truncated, info = env.step([0, 0])
        if terminated or truncated:
            break

    # Then
    assert i == 54
    assert reward < 0


def test_successful_landing(env: MarsLanderEnv):
    # Given
    # https://www.codingame.com/ide/puzzle/mars-lander-episode-1 - Straight landing
    env.reset(
        options={
            "ground": np.array(
                [
                    [0, 100],
                    [1000, 500],
                    [1500, 100],
                    [3000, 100],
                    [5000, 1500],
                    [6999, 1000],
                ]
            ),
            "rover": (2500, 2500, 0, 0, 500, 0, 0),
        }
    )

    # When
    # Boost speed to max
    env.step([0, 1])
    env.step([0, 1])
    env.step([0, 1])
    env.step([0, 1])
    for i in range(70):
        observation, reward, terminated, truncated, info = env.step([0, -1])
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, info = env.step([0, 1])

    # Then
    assert i == 64
    assert reward >= 0
