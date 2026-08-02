import argparse
import glob
import os.path

import gymnasium as gym
from moviepy import VideoFileClip, concatenate_videoclips
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
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="flag to record videos of episodes",
    )
    parser.add_argument(
        "--video-folder",
        default="videos",
        help="path to videos folder",
    )
    args = parser.parse_args()
    return args


def create_full_gif(video_folder: str) -> None:
    clips = [
        VideoFileClip(filename=os.path.join(video_folder, filename))
        for filename in glob.glob("*.mp4", root_dir=video_folder)
    ]
    full_clip = concatenate_videoclips(clips=clips)
    full_clip.write_gif(os.path.join(video_folder, "rl-video-episodes.gif"))


def main() -> None:
    args = parse_args()
    env = gym.make(
        "gymnasium_mars_lander:gymnasium_mars_lander/MarsLander-v1",
        render_mode="rgb_array" if args.record_video else "human",
        episode=args.episode,
    )
    if args.record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
    model = PPO.load(path=args.path, env=env)

    fuel_left = 0
    completion = 0
    test_cases = MARS_LANDER_TEST_CASES[args.episode - 1]

    for test_case in test_cases:
        observation, info = env.reset(
            options={
                "ground": test_case.ground,
                "rover": test_case.rover,
            }
        )

        while True:
            action, _ = model.predict(observation=observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if info["msg"] == "Mission accomplished":
            fuel = info["fuel"]
            completion += 1
        else:
            fuel = 0

        fuel_left += fuel
        print(f"{test_case.name: <30} {info['msg']: <25} {fuel}")

    env.close()
    print(
        f"Fuel left: {fuel_left} - Completion: {100 * completion / len(test_cases):.2f}%"
    )

    if args.record_video:
        create_full_gif(video_folder=args.video_folder)


if __name__ == "__main__":
    main()
