from dataclasses import dataclass


@dataclass
class RoverState:
    """Rover state.

    Attributes:
        x (float): Rover's position along x-axis.
        y (float): Rover's position along y-axis.
        vx (float): Rover's speed along x-axis.
        vy (float): Rover's speed along y-axis.
        fuel (float): Remaining quantity of fuel in liters.
        rotate (float): Angle of rotation of the rover in degrees.
        power (float): Thrust power of the rover.
    """

    x: float
    y: float
    vx: float
    vy: float
    fuel: float
    rotate: float
    power: float

    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def is_within_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> bool:
        return x_min <= self.x < x_max and y_min <= self.y < y_max
