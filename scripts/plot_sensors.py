import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from gymnasium_mars_lander.envs import geometry
from gymnasium_mars_lander.envs.level import MARS_LANDER_TEST_CASES

SCENE_WIDTH = 7000
SCENE_HEIGHT = 3000
DISTANCE_MAX = float(np.linalg.norm([SCENE_WIDTH, SCENE_HEIGHT]))
SENSOR_ANGLES = [-10, -45, -90, -135, -170, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--index",
        default=0,
        type=int,
        help="test case index",
    )
    parser.add_argument(
        "-e",
        "--episode",
        default=2,
        type=int,
        help="episode",
    )
    return parser.parse_args()


def get_target_pos(
    x: float,
    y: float,
    distance: float,
    angle: float,
) -> tuple[float, float]:
    radians = math.radians(angle)
    target_x = x + distance * math.cos(radians)
    target_y = y + distance * math.sin(radians)
    return target_x, target_y


def main() -> None:
    args = parse_args()
    test_case = MARS_LANDER_TEST_CASES[args.episode - 1][args.index]

    fig, ax = plt.subplots()

    ground = np.array(test_case.ground)
    ax.plot(
        ground[:, 0],
        ground[:, 1],
        marker="x",
        color="r",
    )

    rover_pos = test_case.rover[:2]
    ax.plot(
        rover_pos[0],
        rover_pos[1],
        marker="o",
        markersize=5,
        color="b",
    )

    segments = [
        *list(zip(ground[:-1], ground[1:])),
        [(0, 0), (0, SCENE_HEIGHT)],
        [(0, SCENE_HEIGHT), (SCENE_WIDTH, SCENE_HEIGHT)],
        [(SCENE_WIDTH, SCENE_HEIGHT), (SCENE_WIDTH, 0)],
        [(SCENE_WIDTH, 0), (0, 0)],
    ]

    for angle in SENSOR_ANGLES:
        sensor_pos = geometry.get_target_pos(
            pos=rover_pos,
            distance=DISTANCE_MAX,
            angle=angle,
        )
        ax.plot(
            [rover_pos[0], sensor_pos[0]],
            [rover_pos[1], sensor_pos[1]],
            alpha=0.5,
            label=f"{angle}°",
        )

        min_intersection = None
        min_distance = math.inf

        for segment_start, segment_end in segments:
            intersection = geometry.segment_intersection(
                rover_pos,
                sensor_pos,
                segment_start,
                segment_end,
            )
            if intersection is not None:
                d = math.dist(rover_pos, intersection)
                if d < min_distance:
                    min_intersection = intersection
                    min_distance = d

        assert min_intersection is not None
        ax.plot(
            min_intersection[0],
            min_intersection[1],
            marker="o",
            markersize=5,
            color="gray",
        )

    ax.set_title(test_case.name)
    ax.set_xlim(0, SCENE_WIDTH)
    ax.set_ylim(0, SCENE_HEIGHT)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
