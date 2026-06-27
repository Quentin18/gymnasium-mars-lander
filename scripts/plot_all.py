import numpy as np
from matplotlib import pyplot as plt

from gymnasium_mars_lander.envs.level import MARS_LANDER_TEST_CASES

SCENE_WIDTH = 7000
SCENE_HEIGHT = 3000


def main() -> None:
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 8))
    axes = axes.flatten()
    ax_index = 0

    for episode, levels in enumerate(MARS_LANDER_TEST_CASES):
        for level in levels:
            ax = axes[ax_index]
            ax.set_title(f"Episode {episode + 1} - {level.name}")

            ground = np.array(level.ground)

            for x, y in level.starts:
                ax.plot(x, y, marker="o", markersize=5)

            ax.plot(ground[:, 0], ground[:, 1], marker="x", color="r")

            ax.set_xlim(0, SCENE_WIDTH)
            ax.set_ylim(0, SCENE_HEIGHT)
            ax.grid(True)

            ax_index += 1

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
