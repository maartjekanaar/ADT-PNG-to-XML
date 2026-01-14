# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Analysis functions.

from typing import Optional
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from type_aliases_and_data_structures import (
    RotatedRectangle,
)

test_image: Optional[np.ndarray] = None


def show(image: np.ndarray, title: str = "", size: int = 8) -> None:
    plt.figure(figsize=(size, size))
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_rotated_rectangles(
    image: np.ndarray,
    rectangles: list[RotatedRectangle],
    color: tuple[int, int, int] = (255, 0, 0),  # blue
) -> np.ndarray:
    out = image.copy()
    for rectangle in rectangles:
        rectangle_points = cv.boxPoints(rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)
        cv.drawContours(out, [rectangle_points], 0, color, 3)
    return out


def show_all_components(
    image: np.ndarray,
    nodes: list[RotatedRectangle],
    refinement_edges: list[RotatedRectangle],
    countermeasure_edges: list[RotatedRectangle],
    conjunctive_refinement_arcs: list[RotatedRectangle],
) -> None:
    contour_image = image.copy()

    contour_image = draw_rotated_rectangles(
        contour_image, nodes, color=(255, 0, 0)
    )  # blue

    contour_image = draw_rotated_rectangles(
        contour_image, refinement_edges, color=(0, 255, 255)
    )  # yellow

    contour_image = draw_rotated_rectangles(
        contour_image, countermeasure_edges, color=(0, 0, 255)
    )  # red

    contour_image = draw_rotated_rectangles(
        contour_image, conjunctive_refinement_arcs, color=(0, 255, 0)
    )  # green

    show(contour_image, "All components")
