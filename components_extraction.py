# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Functions for components extraction.

import cv2 as cv
import math
import numpy as np

from type_aliases_and_data_structures import (
    Line,
    BoundingRectangle,
    RotatedRectangle,
    RectanglePoints,
)

from parameters import (
    RECTANGLE_EXPANSION_FACTOR,
)

from helper_functions import (
    interior_ink_density,
    draw_lines,
    get_line_endpoints,
    point_near_or_in_rectangle,
)

import analysis_functions
from analysis_functions import (
    draw_rotated_rectangles,
    show,
)


def remove_container_rectangles(
    rectangles: list[RotatedRectangle],
) -> list[RotatedRectangle]:
    """
    Remove rectangles that contain other rectangles from the given <rectangles>.
    Return a list of the remaining rectangles.
    """
    remaining_rectangles: list[RotatedRectangle] = []

    for index1 in range(len(rectangles)):
        rectangle1 = rectangles[index1]
        rectangle1_points = cv.boxPoints(rectangle1)
        rectangle1_points = np.round(rectangle1_points).astype(np.intp)
        contains_other = False

        for index2 in range(len(rectangles)):
            if index1 == index2:
                continue
            rectangle2 = rectangles[index2]
            center2 = rectangle2[0]
            if (
                cv.pointPolygonTest(
                    rectangle1_points, tuple(map(float, center2)), False
                )
                >= 0
            ):
                contains_other = True
                break

        if not contains_other:
            remaining_rectangles.append(rectangle1)

    return remaining_rectangles


def merge_overlapping_rectangles(
    rectangles: list[RotatedRectangle], iou_threshold: float = 0.3
) -> list[RotatedRectangle]:
    """
    Merge overlapping rotated rectangles from the given <rectangles> based on the Intersection over Union (IoU), using the specified <iou_threshold>.
    Return a list of the merged rotated rectangles.
    """
    bounding_rectangles: list[BoundingRectangle] = []
    merged_rectangles: list[RotatedRectangle] = []
    used_rectangles = [False] * len(rectangles)

    for rectangle in rectangles:
        rectangle_points = cv.boxPoints(rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)
        x_left, y_top, width, height = cv.boundingRect(rectangle_points)
        bounding_rectangles.append((x_left, y_top, width, height))

    for index1 in range(len(rectangles)):
        if used_rectangles[index1]:
            continue
        x_left1, y_top1, width1, height1 = bounding_rectangles[index1]
        rectangle1 = (x_left1, y_top1, x_left1 + width1, y_top1 + height1)

        for index2 in range(index1 + 1, len(rectangles)):
            if used_rectangles[index2]:
                continue
            x_left2, y_top2, width2, height2 = bounding_rectangles[index2]
            rectangle2 = (x_left2, y_top2, x_left2 + width2, y_top2 + height2)

            xx1 = max(rectangle1[0], rectangle2[0])
            yy1 = max(rectangle1[1], rectangle2[1])
            xx2 = min(rectangle1[2], rectangle2[2])
            yy2 = min(rectangle1[3], rectangle2[3])
            intersection = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area1 = (rectangle1[2] - rectangle1[0]) * (rectangle1[3] - rectangle1[1])
            area2 = (rectangle2[2] - rectangle2[0]) * (rectangle2[3] - rectangle2[1])
            union = area1 + area2 - intersection

            iou = intersection / (union + 1e-6)

            if iou > iou_threshold:
                used_rectangles[index2] = True

        used_rectangles[index1] = True
        merged_rectangles.append(rectangles[index1])

    return merged_rectangles


def extract_nodes(threshold: np.ndarray) -> list[RotatedRectangle]:
    """
    Extract node regions from the given <threshold> binary image.
    Return a list of the detected node rectangles.
    """
    raw_nodes: list[RotatedRectangle] = []

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    eroded = cv.dilate(threshold, kernel, iterations=1)
    contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)

        # Skip very small contours (noise, small text)
        if area < 200:
            continue

        x_left, y_top, width, height = cv.boundingRect(contour)

        # Skip narrow contours (likely not nodes)
        if width < 40 or height < 20:
            continue

        # Skip small (other shapes) or large (merged groups) rectangles
        rectangle_area = width * height
        if rectangle_area < 2000 or rectangle_area > 100000:
            continue

        # Skip contours with low interior ink density (likely not nodes)
        ink_density = interior_ink_density(threshold, contour)
        if ink_density < 0.01:
            continue

        center = (x_left + width / 2, y_top + height / 2)
        size = (width, height)
        angle = 0.0
        raw_nodes.append((center, size, angle))

    merged_rectangles = merge_overlapping_rectangles(raw_nodes)
    node_rectangles = remove_container_rectangles(merged_rectangles)

    # For analysis:
    if analysis_functions.test_image is not None:
        node_image = draw_rotated_rectangles(
            analysis_functions.test_image, node_rectangles
        )
        show(node_image, "All nodes")

    return node_rectangles


def merge_overlapping_lines_by_orientation(
    image: np.ndarray, lines: list[Line]
) -> list[RotatedRectangle]:
    """
    Merge overlapping given <lines> into rotated rectangles by drawing all lines in the given <image> and finding contours around them based on their orientation.
    Return a list of the rotated rectangles representing the merged lines.
    """
    lines_by_orientation: dict[int, list[Line]] = {}
    merged_lines_by_orientation: list[RotatedRectangle] = []

    for point1, point2 in lines:
        angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        normalised_orientation = int(round(angle / 15))
        lines_by_orientation.setdefault(normalised_orientation, []).append(
            (point1, point2)
        )

    for _, lines in lines_by_orientation.items():
        if not lines:
            continue

        lines_image = draw_lines(image, lines)
        grey = cv.cvtColor(lines_image, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(grey, 250, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(
            threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            rectangle = cv.minAreaRect(contour)
            center, (width, height), angle = rectangle
            merged_lines_by_orientation.append(
                (
                    center,
                    (
                        width * RECTANGLE_EXPANSION_FACTOR,
                        height * RECTANGLE_EXPANSION_FACTOR,
                    ),
                    angle,
                )
            )

    return merge_overlapping_rectangles(merged_lines_by_orientation)


def extract_connectors(
    image: np.ndarray, masked_image: np.ndarray
) -> list[RotatedRectangle]:
    """
    Extract all connectors (refinement edges, countermeasure edges, conjunctive refinement arcs) from the given <masked_image> binary image using HoughLinesP(). Let merge_overlapping_lines() transform them into rotated rectangles on the given <image>.
    Return a list of the rotated rectangles for all connectors.
    """
    raw_connectors: list[Line] = []

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    enhanced = cv.dilate(masked_image, kernel, iterations=1)

    solid_input = cv.medianBlur(enhanced, 3)
    lines = cv.HoughLinesP(
        solid_input,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=10,
        maxLineGap=40,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            raw_connectors.append(((x1, y1), (x2, y2)))

    return merge_overlapping_lines_by_orientation(image, raw_connectors)


def separate_edges(
    threshold: np.ndarray, edges: list[RotatedRectangle]
) -> tuple[list[RotatedRectangle], list[RotatedRectangle]]:
    """
    Separate the given <edges> into refinement edges and countermeasure edges based on their number of contours in the <threshold> image.
    Return a tuple of the rotated rectangles for the refinement edges and the countermeasure edges.
    """
    refinement_edges: list[RotatedRectangle] = []
    countermeasure_edges: list[RotatedRectangle] = []

    for edge in edges:
        rectangle_points = cv.boxPoints(edge)
        rectangle_points = np.round(rectangle_points).astype(np.intp)

        height, width = threshold.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.drawContours(mask, [rectangle_points], -1, (255, 0, 0), thickness=-1)
        region = cv.bitwise_and(threshold, mask)
        x_left, y_top, width, height = cv.boundingRect(rectangle_points)
        region_crop = region[y_top : y_top + height, x_left : x_left + width]
        contours, _ = cv.findContours(
            region_crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 4:
            refinement_edges.append(edge)
        else:
            countermeasure_edges.append(edge)

    # For analysis:
    if analysis_functions.test_image is not None:
        refinement_edges_image = draw_rotated_rectangles(
            analysis_functions.test_image, refinement_edges
        )
        show(refinement_edges_image, "Refinement Edges")
        countermeasure_edges_image = draw_rotated_rectangles(
            analysis_functions.test_image, countermeasure_edges
        )
        show(countermeasure_edges_image, "Countermeasure Edges")

    return refinement_edges, countermeasure_edges


def extract_edges_and_arcs(
    threshold: np.ndarray,
    nodes: list[RotatedRectangle],
    connectors: list[RotatedRectangle],
) -> tuple[list[RotatedRectangle], list[RotatedRectangle], list[RotatedRectangle]]:
    """
    Separate the <connectors> into refinement edges, countermeasure edges, and conjunctive refinement arcs by checking if a connector connects two different <nodes> in the <threshold> image.
    Return a tuple of the rotated rectangles for the refinement edges, countermeasure edges, and conjunctive refinement arcs.
    """
    edges: list[RotatedRectangle] = []
    arcs: list[RotatedRectangle] = []

    node_points: list[RectanglePoints] = [
        np.round(cv.boxPoints(node)).astype(np.intp) for node in nodes
    ]

    for connector in connectors:
        point1, point2 = get_line_endpoints(connector)

        index1 = next(
            (
                index
                for index, node in enumerate(node_points)
                if point_near_or_in_rectangle(point1, node)
            ),
            None,
        )
        index2 = next(
            (
                index
                for index, node in enumerate(node_points)
                if point_near_or_in_rectangle(point2, node)
            ),
            None,
        )

        if index1 is not None and index2 is not None and index1 != index2:
            edges.append(connector)
        else:
            arcs.append(connector)

    # For analysis:
    if analysis_functions.test_image is not None:
        edges_image = draw_rotated_rectangles(analysis_functions.test_image, edges)
        show(edges_image, "Edges")

    refinement_edges, countermeasure_edges = separate_edges(threshold, edges)

    # For analysis:
    if analysis_functions.test_image is not None:
        arcs_image = draw_rotated_rectangles(analysis_functions.test_image, arcs)
        show(arcs_image, "Arcs")

    return refinement_edges, countermeasure_edges, arcs
