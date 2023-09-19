import argparse

import gymnasium as gym
from stable_baselines3 import PPO

from gymnasium_mars_lander.envs.level import MARS_LANDER_TEST_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enjoy Mars Lander model on CodinGame test cases",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="path to model file",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    env = gym.make(
        "gymnasium_mars_lander:gymnasium_mars_lander/MarsLander-v0",
        render_mode="human",
    )
    model = PPO.load(path=args.path, env=env)

    total_reward = 0
    for i, test_case in enumerate(MARS_LANDER_TEST_CASES):
        observation, info = env.reset(options=test_case)

        while True:
            action, _ = model.predict(observation=observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        total_reward += int(reward)
        print(f"Test case {i}: {reward} ({info['msg']})")

    env.close()
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
