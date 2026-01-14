# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Type aliases and data structures for geometric entities and node representation.

from typing import Literal, TypedDict
import cv2 as cv
import numpy as np

Point = tuple[int, int]  # (x, y)
Line = tuple[Point, Point]  # (point1, point2)
BoundingRectangle = tuple[int, int, int, int]  # (x_left, y_top, width, height)

# RotatedRect: tuple[tuple[float, float], tuple[float, float], float] = ((x_center, y_center), (width, height), angle)
RotatedRectangle = cv.typing.RotatedRect

# np.ndarray: array[tuple[float, float], ...] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
RectanglePoints = np.ndarray


class NodeDict(TypedDict, total=True):
    bounding_rectangle: RectanglePoints
    label: str
    child_nodes: list["NodeDict"]
    switch_role: bool
    type: Literal[0, 1]  # 0: attack, 1: defence
    refinement: Literal[0, 1]  # 0: disjunctive (OR), 1: conjunctive (AND)
