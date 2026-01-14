# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Helper functions for geometric computations and image processing.

import numpy as np
import cv2 as cv

from type_aliases_and_data_structures import (
    Point,
    Line,
    RotatedRectangle,
    RectanglePoints,
    NodeDict,
)


def interior_ink_density(
    threshold: np.ndarray, contour: np.ndarray, erode_pixels: int = 4
) -> float:
    """
    Calculate the interior ink density of the given <contour> in the <threshold> binary image. Erode the contour by <erode_pixels> to exclude the borders.
    Return the ink density as a float between 0.0 and 1.0.
    """
    height, width = threshold.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, (255, 0, 0), thickness=-1)

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * erode_pixels + 1, 2 * erode_pixels + 1)
    )
    eroded_mask = cv.erode(mask, kernel, iterations=1)
    inside = cv.bitwise_and(threshold, eroded_mask)
    ink_pixels = cv.countNonZero(inside)
    total_pixels = cv.countNonZero(eroded_mask)

    return ink_pixels / (total_pixels + 1e-6)


def draw_lines(image: np.ndarray, lines: list[Line]) -> np.ndarray:
    """
    Draw the given <lines> in black on a blank white image of the same shape as the input <image>.
    Return the image with the lines drawn.
    """
    output_image = np.ones_like(image) * 255
    for point1, point2 in lines:
        cv.line(output_image, point1, point2, (0, 0, 0), 2)

    return output_image


def get_line_endpoints(rectangle: RotatedRectangle) -> tuple[Point, Point]:
    """
    Get the midpoints of the two shortest sides of the given <rectangle> around a line segment.
    Return a tuple of two Points, representing the midpoints of the shortest sides.
    """
    rectangle_points = cv.boxPoints(rectangle)
    rectangle_points = np.round(rectangle_points).astype(np.intp)

    side_lengths = [
        np.linalg.norm(rectangle_points[0] - rectangle_points[1]),
        np.linalg.norm(rectangle_points[1] - rectangle_points[2]),
        np.linalg.norm(rectangle_points[2] - rectangle_points[3]),
        np.linalg.norm(rectangle_points[3] - rectangle_points[0]),
    ]
    order = np.argsort(side_lengths)
    index1 = order[0]
    pair1 = {index1, (index1 + 1) % 4}

    index2 = order[1]
    for index in order[1:]:
        pair2 = {index, (index + 1) % 4}
        if pair1.isdisjoint(pair2):
            index2 = index
            break

    def midpoint(index1: int, index2: int) -> Point:
        return (
            int(round((rectangle_points[index1][0] + rectangle_points[index2][0]) / 2)),
            int(round((rectangle_points[index1][1] + rectangle_points[index2][1]) / 2)),
        )

    point1 = midpoint(index1, (index1 + 1) % 4)
    point2 = midpoint(index2, (index2 + 1) % 4)
    return point1, point2


def point_near_or_in_rectangle(
    point: tuple[float, float], rectangle_points: RectanglePoints, threshold: int = 15
) -> bool:
    """
    Check if the given <point> is near or inside the rotated rectangle with the given <rectangle_points>.
    Return True if the point is inside or within <threshold> distance of the rectangle, False otherwise.
    """
    distance = cv.pointPolygonTest(
        rectangle_points, tuple(map(float, point)), measureDist=True
    )

    return distance >= -threshold


def is_same_node(node1: NodeDict, node2: NodeDict) -> bool:
    """
    Check if the two given node dictionaries, <node1> and <node2>, represent the same node based on their label and bounding rectangle.
    Return True if both nodes have the same label and bounding rectangle, False otherwise.
    """
    return node1["label"] == node2["label"] and np.array_equal(
        node1["bounding_rectangle"], node2["bounding_rectangle"]
    )
