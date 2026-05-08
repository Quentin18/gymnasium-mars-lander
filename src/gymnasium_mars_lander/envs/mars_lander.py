import dataclasses
import math
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
from gymnasium_mars_lander.envs.utils import minmax_scale

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
        - Angle between -15° and +15°
        - Thrust power between -1 and +1

    Observation Space:
        - Distances in 6 directions
        - Rover horizontal and vertical speed, angle, power
        - Horizontal and vertical distances relative to the middle of the landing area
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        episode: int = 2,
        start: int = -1,
        eval_env: bool = False,
        sequential_maps: bool = False,
    ) -> None:
        self.episode = episode
        self.start = start
        self.eval_env = eval_env
        self.sequential_maps = sequential_maps
        self.test_index = -1

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
        self.distance_max = float(np.linalg.norm([self.scene_width, self.scene_height]))
        self.sensor_angles = [-10, -45, -90, -135, -170, 90]

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len(self.sensor_angles) + 6,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None
        self.rover_img = None

    def _generate_random_input(self) -> tuple[np.ndarray, RoverState]:
        test_cases = MARS_LANDER_TEST_CASES[self.episode - 1]

        if self.sequential_maps:
            self.test_index = (self.test_index + 1) % len(test_cases)
        else:
            self.test_index = self.np_random.choice(len(test_cases))

        test_case = test_cases[self.test_index]

        ground = np.array(test_case["ground"], dtype=self.observation_space.dtype)
        rover = RoverState(*test_case["rover"])
        if self.start >= 0:
            rover.x, rover.y = test_case["starts"][self.start]
            rover.vx = rover.vy = rover.rotate = rover.power = 0

        # flip left-right
        if not self.eval_env and self.np_random.random() < 0.5:
            ground[:, 0] = self.scene_width - ground[:, 0]
            ground = ground[::-1]
            rover.x = self.scene_width - rover.x
            rover.vx *= -1
            rover.rotate *= -1

        # shift rover randomly
        rover.y += self.np_random.uniform(-50, 50)
        rover.x += self.np_random.uniform(-50, 50)

        # shift ground randomly
        ground[:, 0] += self.np_random.uniform(-50, 50)
        ground[:, 1] += self.np_random.uniform(-50, 50)

        # clip values
        ground[:, 0] = np.clip(ground[:, 0], 0, self.scene_width)
        ground[:, 1] = np.clip(ground[:, 1], 0, self.scene_height)
        ground[0, 0] = 0
        ground[-1, 0] = self.scene_width - 1
        rover.x = np.clip(rover.x, 0, self.scene_width)
        rover.y = np.clip(rover.y, 0, self.scene_height)

        return ground, rover

    def _get_sensors_intersection(self) -> tuple[np.ndarray, np.ndarray]:
        points = []
        distances = []

        segments = [
            *list(zip(self.ground[:-1], self.ground[1:])),
            [(0, 0), (0, self.scene_height)],
            [(0, self.scene_height), (self.scene_width, self.scene_height)],
            [(self.scene_width, self.scene_height), (self.scene_width, 0)],
            [(self.scene_width, 0), (0, 0)],
        ]

        for angle in self.sensor_angles:
            sensor_pos = geometry.get_target_pos(
                pos=self.rover.position(),
                distance=self.distance_max,
                angle=angle,
            )
            min_point = (math.nan, math.nan)
            min_distance = math.inf

            for segment_start, segment_end in segments:
                intersection = geometry.segment_intersection(
                    self.rover.position(),
                    sensor_pos,
                    segment_start,
                    segment_end,
                )
                if intersection is None:
                    continue

                d = math.dist(self.rover.position(), intersection)
                if d < min_distance:
                    min_point = intersection
                    min_distance = d

            points.append(min_point)
            distances.append(0.0 if math.isinf(min_distance) else min_distance)

        return (
            np.array(points, dtype=self.observation_space.dtype),
            np.array(distances, dtype=self.observation_space.dtype),
        )

    def _get_rover_obs(self) -> np.ndarray:
        return np.array(
            [
                minmax_scale(self.rover.vx, -self.speed_max, self.speed_max),
                minmax_scale(self.rover.vy, -self.speed_max, self.speed_max),
                minmax_scale(self.rover.rotate, self.rotate_min, self.rotate_max),
                minmax_scale(self.rover.power, self.power_min, self.power_max),
            ],
            dtype=self.observation_space.dtype,
        )

    def _get_landing_area_center(self) -> np.ndarray:
        return self.ground[self.landing_area].mean(axis=0)

    def _get_landing_area_obs(self) -> np.ndarray:
        target_x, target_y = self.landing_area_center
        return np.array(
            [
                minmax_scale(
                    target_x - self.rover.x,
                    -self.distance_max,
                    self.distance_max,
                ),
                minmax_scale(
                    target_y - self.rover.y,
                    -self.distance_max,
                    self.distance_max,
                ),
            ],
            dtype=self.observation_space.dtype,
        )

    def _get_obs(self) -> ObsType:
        _, sensor_distances = self._get_sensors_intersection()
        return np.concat(
            [
                2 * sensor_distances / self.distance_max - 1,
                self._get_rover_obs(),
                self._get_landing_area_obs(),
            ],
            dtype=self.observation_space.dtype,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "ground" in options and "rover" in options:
            self.ground = np.array(
                options["ground"],
                dtype=self.observation_space.dtype,
            )
            self.rover = RoverState(*options["rover"])
        else:
            self.ground, self.rover = self._generate_random_input()

        self.landing_area = geometry.find_flat_segment(polygon=self.ground)
        self.landing_area_center = self._get_landing_area_center()

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

        prev_state = dataclasses.replace(self.rover)
        prev_dist = math.dist(self.rover.position(), self.landing_area_center)

        rotate, power = self._convert_action_to_rotate_power(action=action)
        self._update_state(rotate=rotate, power=power)

        observation = self._get_obs()
        dist = math.dist(self.rover.position(), self.landing_area_center)
        reward = 0.01 if dist < prev_dist else 0
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
            observation[: len(self.sensor_angles)] = 0
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
                self.rover.y <= self.ground[self.landing_area[0]][1]
                and self.ground[self.landing_area[0]][0]
                <= self.rover.x
                <= self.ground[self.landing_area[1]][0]
            )
            has_no_angle = abs(prev_state.rotate) <= 15 and abs(self.rover.rotate) <= 15
            has_low_speed = abs(self.rover.vy) <= 40 and abs(self.rover.vx) <= 20
            mission_completed = on_flat_ground and has_no_angle and has_low_speed

            if not mission_completed:
                info["msg"] = "Rover has been destroyed"
                if on_flat_ground:
                    reward = -50
                elif has_no_angle and has_low_speed:
                    reward = -75
                else:
                    reward = -100
            else:
                info["msg"] = "Mission accomplished"
                reward = 200 + self.rover.fuel

            terminated = True
            observation[: len(self.sensor_angles)] = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        if self.window is None and self.render_mode == "human":
            pygame.init()  # pylint: disable=no-member
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
                angle=-self.rover.rotate,
            ),
            dest=(
                self.rover.x // SCALE_FACTOR - self.rover_img.get_width() // 2,
                self.rover.y // SCALE_FACTOR,
            ),
        )

        canvas = pygame.transform.flip(surface=canvas, flip_x=False, flip_y=True)

        # Draw rover state
        state_dict = {
            "position": self.rover.x,
            "altitude": self.rover.y,
            "horizontal speed": self.rover.vx,
            "vertical speed": self.rover.vy,
            "fuel": self.rover.fuel,
            "rotate": self.rover.rotate,
            "power": self.rover.power,
        }
        for i, (key, value) in enumerate(state_dict.items()):
            text_surface = self.font.render(
                f"{key.upper():<20} {int(value)}",
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
            pygame.quit()  # pylint: disable=no-member
