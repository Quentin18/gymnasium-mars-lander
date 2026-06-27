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
        "gymnasium_mars_lander:gymnasium_mars_lander/MarsLander-v0",
        render_mode="rgb_array" if args.record_video else "human",
    )
    if args.record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
    model = PPO.load(path=args.path, env=env)

    total_fuel = 0
    for test_case in MARS_LANDER_TEST_CASES[args.episode - 1]:
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

        total_fuel += info["fuel"]
        print(f"{test_case.name: <30} {info['msg']: <25} {info['fuel']}")

    env.close()
    print("Total fuel:", int(total_fuel))

    if args.record_video:
        create_full_gif(video_folder=args.video_folder)


if __name__ == "__main__":
    main()
