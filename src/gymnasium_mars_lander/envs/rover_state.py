import numpy as np


class RoverState:
    """Encapsulate the Mars Lander state.

    The state is represented as a contiguous array but
    getters and setters are provided to ease its manipulation.

    Attributes:
        x (float): Rover's position along x-axis.
        y (float): Rover's position along y-axis.
        vx (float): Rover's speed along x-axis.
        vy (float): Rover's speed along y-axis.
        fuel (float): Remaining quantity of fuel in liters.
        rotate (float): Angle of rotation of the rover in degrees.
        power (float): Thrust power of the rover.
    """

    def __init__(self, state: np.ndarray | list[float] | tuple[float]) -> None:
        """Initialize the rover state.

        Args:
            state: Array of 7 physical measurements represented as floats.
        """
        self._state = np.array(state, dtype=np.float32)
        assert self._state.shape == (7,), "invalid rover state shape"

    @property
    def x(self) -> float:
        return self._state[0]

    @property
    def y(self) -> float:
        return self._state[1]

    @property
    def vx(self) -> float:
        return self._state[2]

    @property
    def vy(self) -> float:
        return self._state[3]

    @property
    def fuel(self) -> float:
        return self._state[4]

    @property
    def rotate(self) -> float:
        return self._state[5]

    @property
    def power(self) -> float:
        return self._state[6]

    @x.setter
    def x(self, value: float) -> None:
        self._state[0] = value

    @y.setter
    def y(self, value: float) -> None:
        self._state[1] = value

    @vx.setter
    def vx(self, value: float) -> None:
        self._state[2] = value

    @vy.setter
    def vy(self, value: float) -> None:
        self._state[3] = value

    @fuel.setter
    def fuel(self, value: float) -> None:
        self._state[4] = value

    @rotate.setter
    def rotate(self, value: float) -> None:
        self._state[5] = value

    @power.setter
    def power(self, value: float) -> None:
        self._state[6] = value

    @staticmethod
    def names() -> list[str]:
        return [
            "position",
            "altitude",
            "horizontal speed",
            "vertical speed",
            "fuel",
            "rotate",
            "power",
        ]

    def numpy(self) -> np.ndarray:
        return self._state

    def is_within_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> bool:
        return x_min <= self.x < x_max and y_min <= self.y < y_max
