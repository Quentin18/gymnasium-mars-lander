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

    for _ in range(args.n_samples):
        ground = np.array(
            MARS_LANDER_TEST_CASES[args.index]["ground"],
            dtype=np.float64,
        )
        rover = RoverState(MARS_LANDER_TEST_CASES[args.index]["rover"])

        # flip left-right
        if args.flip:
            ground[:, 0] = SCENE_WIDTH - ground[:, 0]
            ground = ground[::-1]
            rover.x = SCENE_WIDTH - rover.x
            rover.vx *= -1
            rover.rotate *= -1

        # shift rover randomly
        if args.shift_rover:
            rover.y += np.random.random() * 2 * 50 - 50
            rover.x += np.random.random() * 2 * 30 - 30

        # shift ground randomly
        if args.shift_ground:
            ground[:, 0] += np.random.random() * 2 * 50 - 50
            if args.index != 4:
                ground[:, 1] -= np.random.random() * 50
            else:
                ground[:, 1] += np.random.random() * 2 * 50 - 50

        plt.plot(ground[:, 0], ground[:, 1], marker="x", color="r", linewidth=0.2)
        plt.plot(rover.x, rover.y, marker="o", markersize=10)

    plt.xlim(0, SCENE_WIDTH)
    plt.ylim(0, SCENE_HEIGHT)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
