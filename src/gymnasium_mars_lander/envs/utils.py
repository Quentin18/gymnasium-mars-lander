def minmax_scale(x: float, x_min: float, x_max: float) -> float:
    return 2 * (x - x_min) / (x_max - x_min) - 1
