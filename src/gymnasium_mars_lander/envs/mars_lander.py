import os
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

from gymnasium_mars_lander.envs import geometry
from gymnasium_mars_lander.envs.level import MARS_LANDER_TEST_CASES
from gymnasium_mars_lander.envs.rover_state import RoverState

BACKGROUND_COLOR = (0, 0, 0)
GROUND_COLOR = (255, 0, 0)
SCALE_FACTOR = 8

FONT_NAME = "Monospace"
FONT_COLOR = (255, 255, 255)
FONT_SIZE = 100

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROVER_IMG_FILENAME = os.path.join(DIRNAME, "assets", "rover.png")


class MarsLanderEnv(gym.Env):
    """Environment for the Mars Lander CodinGame optimization game.

    Reference: https://www.codingame.com/multiplayer/optimization/mars-lander

    Action Space:
        Continuous values for angle of rotation and thrust power

    Observation Space:
        - ground: Surface of Mars as a broken line. 30 pairs of 2d points (x, y).
        - landing: Start and end points of the flat section.
        - rover: Rover state. 7 float numbers.

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None) -> None:
        self.gravity = 3.711  # meters/sec^2
        self.scene_width = 7000  # meters
        self.scene_height = 3000  # meters
        self.speed_max = 500  # meters/sec
        self.fuel_max = 2000  # liters
        self.rotate_min = -90  # degrees
        self.rotate_max = 90  # degrees
        self.rotate_max_step = 15  # degrees
        self.power_min = 0
        self.power_max = 4
        self.power_max_step = 1

        self.observation_space = spaces.Dict(
            {
                "ground": spaces.Box(low=0, high=1, shape=(30, 2), dtype=np.float32),
                "landing": spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32),
                "rover": spaces.Box(
                    low=np.array([0, 0, -1, -1, 0, -1, 0]),
                    high=np.array([1, 1, 1, 1, 1, 1, 1]),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None
        self.rover_img = None

    def _get_obs(self) -> ObsType:
        observation = {
            "ground": (self.ground / [self.scene_width, self.scene_height]).astype(
                np.float32
            ),
            "landing": (
                self.ground[self.landing_area] / [self.scene_width, self.scene_height]
            ).astype(np.float32),
            "rover": (
                np.rint(self.rover.numpy())
                / [
                    self.scene_width,
                    self.scene_height,
                    self.speed_max,
                    self.speed_max,
                    self.fuel_max,
                    self.rotate_max,
                    self.power_max,
                ]
            ).astype(np.float32),
        }
        return observation

    def _generate_random_input(self, evaluate: bool) -> tuple[np.ndarray, RoverState]:
        test_index = self.np_random.choice(np.arange(len(MARS_LANDER_TEST_CASES)))
        ground = np.array(
            MARS_LANDER_TEST_CASES[test_index]["ground"],
            dtype=np.float32,
        )
        rover = RoverState(MARS_LANDER_TEST_CASES[test_index]["rover"])

        # flip left-right
        if not evaluate and self.np_random.random() < 0.5:
            ground[:, 0] = self.scene_width - ground[:, 0]
            ground = ground[::-1]
            rover.x = self.scene_width - rover.x
            rover.vx *= -1
            rover.rotate *= -1

        # add fuel randomly
        if not evaluate:
            rover.fuel += self.np_random.random() * 150

        # shift rover randomly
        rover.y += self.np_random.random() * 2 * 50 - 50
        rover.x += self.np_random.random() * 2 * 30 - 30

        # shift ground randomly
        ground[:, 0] += self.np_random.random() * 2 * 50 - 50
        if test_index != 4:
            ground[:, 1] -= self.np_random.random() * 50
        else:
            ground[:, 1] += self.np_random.random() * 2 * 50 - 50

        # clip values
        ground[:, 0] = np.clip(ground[:, 0], 0, self.scene_width)
        ground[:, 1] = np.clip(ground[:, 1], 0, self.scene_height)
        ground[0, 0] = 0
        ground[-1, 0] = self.scene_width - 1
        rover.x = np.clip(rover.x, 0, self.scene_width)
        rover.y = np.clip(rover.y, 0, self.scene_height)
        rover.fuel = np.clip(rover.fuel, 0, self.fuel_max)

        return ground, rover

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # eval mode to use CG test cases
        evaluate = bool(None if options is None else options.get("eval"))

        if options is not None and "ground" in options and "rover" in options:
            self.ground = np.array(options["ground"], dtype=np.float32)
            self.rover = RoverState(options["rover"])
        else:
            self.ground, self.rover = self._generate_random_input(evaluate=evaluate)

        self.ground = geometry.convert_to_fixed_length_polygon(
            polygon=self.ground,
            n=self.observation_space["ground"].shape[0],
        )
        self.landing_area = geometry.find_flat_segment(polygon=self.ground)

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _convert_action_to_rotate_power(self, action: np.ndarray) -> tuple[int, int]:
        rotate, power = action
        assert -1 <= rotate <= 1
        assert -1 <= power <= 1

        rotate = np.rint(rotate * self.rotate_max_step)
        power = -1 if power < -1 / 3 else 0 if power < 1 / 3 else 1

        return rotate, power

    def _update_state(self, rotate: int, power: int) -> None:
        # Update rotation. Value of the previous turn +/- 15°.
        self.rover.rotate += np.rint(rotate)
        self.rover.rotate = np.clip(self.rover.rotate, self.rotate_min, self.rotate_max)

        # Update engine power. Value of the previous turn +/- 1.
        self.rover.power += np.rint(power)
        self.rover.power = np.clip(self.rover.power, self.power_min, self.power_max)

        # Tank content. For a thrust power of T, T liters of fuel are consumed.
        self.rover.fuel -= abs(self.rover.power)

        # Newton's second law projected along X and Y
        new_vy = (
            self.rover.vy
            + np.cos(np.radians(self.rover.rotate)) * self.rover.power
            - self.gravity
        )
        self.rover.y += (new_vy + self.rover.vy) / 2
        self.rover.vy = new_vy

        new_vx = (
            self.rover.vx - np.sin(np.radians(self.rover.rotate)) * self.rover.power
        )
        self.rover.x += (new_vx + self.rover.vx) / 2
        self.rover.vx = new_vx

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        prev_rotate = self.rover.rotate
        rotate, power = self._convert_action_to_rotate_power(action=action)
        self._update_state(rotate=rotate, power=power)

        observation = self._get_obs()
        reward = 1
        terminated = False
        info = {}

        if not self.rover.is_within_bounds(
            x_min=0,
            x_max=self.scene_width,
            y_min=0,
            y_max=self.scene_height,
        ):
            info["msg"] = "Rover is running away"
            reward = -150
            terminated = True
        elif self.rover.fuel < 0:
            info["msg"] = "Tank is empty"
            reward = -150
            terminated = True
        elif geometry.is_inside_ground(
            ground=self.ground,
            x=self.rover.x,
            y=self.rover.y,
            y_max=self.scene_height,
        ):
            on_flat_ground = (
                self.ground[self.landing_area[0]][0]
                <= self.rover.x
                <= self.ground[self.landing_area[1]][0]
            )
            has_no_angle = (
                int(abs(prev_rotate)) <= 15 and int(abs(self.rover.rotate)) <= 15
            )
            has_low_speed = (
                int(abs(self.rover.vy)) <= 40 and int(abs(self.rover.vx)) <= 20
            )
            mission_completed = on_flat_ground and has_no_angle and has_low_speed

            if not mission_completed:
                info["msg"] = "Rover has been destroyed"
                reward = (
                    -50
                    if on_flat_ground
                    else -75
                    if has_no_angle and has_low_speed
                    else -100
                )
            else:
                info["msg"] = "Mission accomplished"
                reward = self.rover.fuel - self.rover.power**2
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.scene_width // SCALE_FACTOR, self.scene_height // SCALE_FACTOR)
            )
            pygame.display.set_caption("Mars Lander")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE // SCALE_FACTOR)

        if self.rover_img is None:
            self.rover_img = pygame.transform.flip(
                surface=pygame.transform.scale_by(
                    surface=pygame.image.load(ROVER_IMG_FILENAME),
                    factor=0.4 / SCALE_FACTOR,
                ),
                flip_x=False,
                flip_y=True,
            )

        canvas = pygame.Surface(
            (self.scene_width // SCALE_FACTOR, self.scene_height // SCALE_FACTOR)
        )
        canvas.fill(BACKGROUND_COLOR)

        # Draw ground
        pygame.draw.lines(
            surface=canvas,
            color=GROUND_COLOR,
            closed=False,
            points=(self.ground // SCALE_FACTOR).tolist(),
        )

        # Draw rover
        canvas.blit(
            source=pygame.transform.rotate(
                surface=self.rover_img,
                angle=self.rover.rotate,
            ),
            dest=(
                self.rover.x // SCALE_FACTOR - self.rover_img.get_width() // 2,
                self.rover.y // SCALE_FACTOR,
            ),
        )

        canvas = pygame.transform.flip(surface=canvas, flip_x=False, flip_y=True)

        # Draw rover state
        for i, (name, value) in enumerate(zip(self.rover.names(), self.rover.numpy())):
            text_surface = self.font.render(
                f"{name.upper():<20} {int(value)}",
                True,
                FONT_COLOR,
            )
            canvas.blit(source=text_surface, dest=(0, i * self.font.get_height()))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
