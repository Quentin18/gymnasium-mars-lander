import argparse

import matplotlib.pyplot as plt
import numpy as np

from gymnasium_mars_lander.envs.level import MARS_LANDER_TEST_CASES
from gymnasium_mars_lander.envs.rover_state import RoverState

SCENE_WIDTH = 7000
SCENE_HEIGHT = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Mars Lander surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help="test case index",
    )
    parser.add_argument(
        "-e",
        "--episode",
        default=2,
        type=int,
        help="episode",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="number of test case samples to generate",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="flip left-right",
    )
    parser.add_argument(
        "--shift-rover",
        action="store_true",
        help="shift rover",
    )
    parser.add_argument(
        "--shift-ground",
        action="store_true",
        help="shift ground",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    test_case = MARS_LANDER_TEST_CASES[args.episode - 1][args.index]
    print("Test case:", test_case.name)

    for _ in range(args.n_samples):
        ground = np.array(test_case.ground, dtype=np.float64)
        rover = RoverState(*test_case.rover)

        # flip left-right
        if args.flip:
            ground[:, 0] = SCENE_WIDTH - ground[:, 0]
            ground = ground[::-1]
            rover.x = SCENE_WIDTH - rover.x
            rover.vx *= -1
            rover.rotate *= -1

        # shift rover randomly
        if args.shift_rover:
            rover.y += np.random.uniform(-50, 50)
            rover.x += np.random.uniform(-50, 50)

        # shift ground randomly
        if args.shift_ground:
            ground[:, 0] += np.random.uniform(-50, 50)
            ground[:, 1] += np.random.uniform(-50, 50)

        plt.plot(ground[:, 0], ground[:, 1], marker="x", color="r", linewidth=0.2)
        plt.plot(rover.x, rover.y, marker="o", markersize=10)

    plt.xlim(0, SCENE_WIDTH)
    plt.ylim(0, SCENE_HEIGHT)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
