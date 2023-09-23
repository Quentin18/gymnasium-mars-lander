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
    parser.add_argument(
        "-e",
        "--episode",
        default=2,
        type=int,
        help="episode",
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
    for test_case in MARS_LANDER_TEST_CASES[args.episode - 1]:
        observation, info = env.reset(options=test_case)

        while True:
            action, _ = model.predict(observation=observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        total_reward += reward
        print(f"{test_case['name']: <30} {info['msg']: <25} {int(reward)}")

    env.close()
    print("Total reward:", int(total_reward))


if __name__ == "__main__":
    main()
