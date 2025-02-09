import math

import numpy as np


def l2dist(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance between two points.

    Args:
        x: First point.
        y: Second point.

    Returns:
        Euclidean distance between x and y.

    """
    return np.linalg.norm(x - y, ord=2)


def convert_to_fixed_length_polygon(polygon: np.ndarray, n: int = 30) -> np.ndarray:
    """Represent a polygon with a certain number of nodes
    by breaking down the longest segments.

    Args:
        polygon: Array of 2D coordinates (x, y).
        n: Expected number of nodes.

    Returns: Polygon of size n.

    """
    if len(polygon) > n:
        raise ValueError(f"The polygon has more than {n} nodes")

    while len(polygon) < n:
        index_longest = np.argmax([l2dist(x, y) for x, y in zip(polygon, polygon[1:])])
        intermediate_point = [
            (polygon[index_longest][0] + polygon[index_longest + 1][0]) / 2,
            (polygon[index_longest][1] + polygon[index_longest + 1][1]) / 2,
        ]
        polygon = np.insert(polygon, index_longest + 1, intermediate_point, 0)
    return polygon


def find_flat_segment(polygon: np.ndarray) -> list[int]:
    """Find start and end indexes of a horizontal section in the polygon.

    This section can be made of several consecutive segments.
    If the polygon contains several flat sections,
    this function only identifies the first one.

    Args:
        polygon: Array of 2D coordinates (x, y).

    Returns: Indexes of start and end points of the flat section.

    """
    start = -1
    end = -1
    for i, (a, b) in enumerate(zip(polygon[:-1], polygon[1:])):
        if a[1] == b[1]:
            if start < 0:
                start = i
            end = i + 1
    return [start, end]


def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if points a, b and c are listed in a counterclockwise order.

    Reference: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Args:
        a: First point.
        b: Second point.
        c: Third point.

    Returns: True if abc is counterclockwise, False otherwise.

    """
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def segment_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    """Check if two segments, [a, b] and [c, d], intersect each other.

    Reference: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Args:
        a: First point of first segment.
        b: Second point of first segment.
        c: First point of second segment.
        d: Second point of second segment.

    Returns: True if the two segments intersect, False otherwise.

    """
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def is_inside_ground(ground: np.ndarray, x: float, y: float, y_max: float) -> bool:
    """Check if a point is inside the ground.

    Inspired by
    https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

    Args:
        ground: List of points representing the ground.
        x: Point position along x-axis.
        y: Point position along y-axis.
        y_max: Maximum height.

    Returns: True if the point is inside the ground, False otherwise.

    """
    pos = np.array([x, y])
    top = np.array([x, y_max])
    intersection_count = 0
    for start, end in zip(ground, ground[1:]):
        if segment_intersect(start, end, pos, top):
            intersection_count += 1
    return bool(intersection_count % 2)


def get_target_pos(
    pos: tuple[float, float],
    distance: float,
    angle: float,
) -> tuple[float, float]:
    x, y = pos
    radians = math.radians(angle)
    target_x = x + distance * math.cos(radians)
    target_y = y + distance * math.sin(radians)
    return target_x, target_y


def segment_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> tuple[float, float] | None:
    denominator = (c[0] - d[0]) * (a[1] - b[1]) - (c[1] - d[1]) * (a[0] - b[0])
    if denominator == 0:
        return None

    t = ((c[0] - a[0]) * (a[1] - b[1]) - (c[1] - a[1]) * (a[0] - b[0])) / denominator
    u = ((c[0] - a[0]) * (c[1] - d[1]) - (c[1] - a[1]) * (c[0] - d[0])) / denominator

    if 0 <= t <= 1 and u >= 0:
        intersect_x = c[0] + t * (d[0] - c[0])
        intersect_y = c[1] + t * (d[1] - c[1])
        return intersect_x, intersect_y

    return None
