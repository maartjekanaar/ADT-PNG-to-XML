# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).

from ADTCreate.ADT import ADT, initADT
from typing import Literal, TypedDict
import cv2 as cv
import math
import numpy as np
import os
import pytesseract
import re
import sys

### Type aliases and data structures: ###

Point = tuple[int, int]  # (x, y)
Line = tuple[Point, Point]  # (point1, point2)
BoundingRectangle = tuple[int, int, int, int]  # (x_left, y_top, width, height)

# RotatedRect: tuple[tuple[float, float], tuple[float, float], float] = ((x_center, y_center), (width, height), angle)
RotatedRectangle = cv.typing.RotatedRect

# np.ndarray: array[tuple[float, float], ...] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
RectanglePoints = np.ndarray

RECTANGLE_EXPANSION_FACTOR = 1.25


class NodeDict(TypedDict, total=True):
    bounding_rectangle: RectanglePoints
    label: str
    child_nodes: list["NodeDict"]
    switch_role: bool
    type: Literal[0, 1]  # 0: attack, 1: defence
    refinement: Literal[0, 1]  # 0: disjunctive (OR), 1: conjunctive (AND)


### Helper functions: ###


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


### Main processing functions: ###


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

    return node_rectangles


def merge_overlapping_lines_by_orientation(
    image: np.ndarray, lines: list[Line]
) -> list[RotatedRectangle]:
    """
    Merge overlapping given <lines> into rotated rectangles by drawing all lines in the given <image> and finding contours around them.
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
    Extract all connectors (refinement edges, countermeasure edges, conjunctive refinement arcs) from the given <masked_image> binary image. Let merge_overlapping_lines() transform them into rotated rectangles on the given <image>.
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

    return refinement_edges, countermeasure_edges


def extract_edges_and_arcs(
    threshold: np.ndarray,
    nodes: list[RotatedRectangle],
    connectors: list[RotatedRectangle],
) -> tuple[list[RotatedRectangle], list[RotatedRectangle], list[RotatedRectangle]]:
    """
    Separate the <connectors> into refinement edges, countermeasure edges, and conjunctive refinement arcs using the given <nodes> and <threshold> image.
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

    refinement_edges, countermeasure_edges = separate_edges(threshold, edges)

    return refinement_edges, countermeasure_edges, arcs


def create_node_objects_with_text(
    image: np.ndarray, nodes: list[RotatedRectangle]
) -> list[NodeDict]:
    """
    Create node objects from the given <nodes> and add extracted text labels using OCR on the <image>.
    Return a list of NodeDict with their bounding rectangles and labels.
    """
    node_objects_with_text: list[NodeDict] = []

    for rotated_rectangle in nodes:
        rectangle_points = cv.boxPoints(rotated_rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)

        x_left, y_top, width, height = cv.boundingRect(rectangle_points)
        x1 = max(0, x_left)
        y1 = max(0, y_top)
        x2 = min(image.shape[1], x_left + width)
        y2 = min(image.shape[0], y_top + height)

        roi = image[y1:y2, x1:x2]

        roi_grey = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        _, roi_threshold = cv.threshold(roi_grey, 200, 255, cv.THRESH_BINARY)

        raw_text: str = pytesseract.image_to_string(
            roi_threshold, lang="eng", config="--psm 6"
        )
        text = raw_text.strip()

        # Replace multiple whitespaces with single space
        text = re.sub(r"\s+", " ", text)
        # Only keep characters allowed by the XSD
        text = re.sub(r"[^0-9A-Za-z\s\?\!\-_\']", "", text)

        if not text or len(text) < 2:
            text = "node"

        node_objects_with_text.append(
            {
                "bounding_rectangle": rectangle_points,
                "label": text,
                "child_nodes": [],
                "switch_role": False,
                "type": 0,
                "refinement": 0,
            }
        )

    return node_objects_with_text


def detect_edge(
    parent_node: NodeDict, child_node: NodeDict, edges: list[RotatedRectangle]
) -> bool:
    """
    Detect whether there is an edge between the given <parent_node> and <child_node> using the list of <edges>.
    Return True if an edge exists between them, False otherwise.
    """
    for edge in edges:
        point1, point2 = get_line_endpoints(edge)
        point_high = point1 if point1[1] < point2[1] else point2
        point_low = point2 if point1[1] < point2[1] else point1

        if point_near_or_in_rectangle(
            point_high, parent_node["bounding_rectangle"]
        ) and point_near_or_in_rectangle(point_low, child_node["bounding_rectangle"]):
            return True

    return False


def assign_child_nodes_to_node_objects(
    node_objects: list[NodeDict],
    refinement_edges: list[RotatedRectangle],
    countermeasure_edges: list[RotatedRectangle],
) -> NodeDict | None:
    """
    Assign child nodes to each parent node in the given <node_objects> based on the <refinement_edges> and <countermeasure_edges>. Assign 'switch_role' accordingly (False for child nodes connected by refinement edges, True for child nodes connected by countermeasure edges).
    Return the root node of the NodeDict tree.
    """
    unassigned_child_nodes = node_objects.copy()

    for parent_node in node_objects:
        child_nodes: list[NodeDict] = []
        for child_node in list(unassigned_child_nodes):
            append_child = False
            if is_same_node(parent_node, child_node):
                continue
            if detect_edge(parent_node, child_node, refinement_edges):
                child_node["switch_role"] = False
                append_child = True
            elif detect_edge(parent_node, child_node, countermeasure_edges):
                child_node["switch_role"] = True
                append_child = True
            if append_child:
                child_nodes.append(child_node)
                for index, node in enumerate(unassigned_child_nodes):
                    if is_same_node(node, child_node):
                        unassigned_child_nodes.pop(index)
                        break
        child_nodes.sort(key=lambda node: node["bounding_rectangle"][:, 0].min())
        parent_node["child_nodes"] = child_nodes

    root_node: NodeDict | None = None
    for node in unassigned_child_nodes:
        if node["child_nodes"] != []:
            root_node = node
            break
    if root_node is None:
        return None

    updated_root_node = next(
        node for node in node_objects if is_same_node(node, root_node)
    )

    return updated_root_node


def assign_type_to_node_objects(root_node: NodeDict) -> NodeDict:
    """
    Assign attack(0)/defence(1) types to all node objects in the tree rooted at the given <root_node>, based on 'switch_role' attributes. The root node is assigned type 0 and types propagate down the tree.
    Return the root node of the updated NodeDict tree.
    """
    root_node["type"] = 0
    queue: list[NodeDict] = [root_node]

    while queue:
        parent_node = queue.pop(0)
        parent_type = parent_node["type"]
        child_nodes = parent_node["child_nodes"]
        for child_node in child_nodes:
            if not child_node["switch_role"]:
                child_type = parent_type
            elif child_node["switch_role"]:
                child_type = 0 if parent_type == 1 else 1
            else:
                continue
            child_node["type"] = child_type
            queue.append(child_node)

    return root_node


def detect_conjunctive_refinement_arc(
    parent_node: NodeDict,
    child_nodes: list[NodeDict],
    conjunctive_refinement_arcs: list[RotatedRectangle],
) -> bool:
    """
    Detect whether there is a conjunctive refinement arc in the rectangular area between the given <parent_node> and <child_nodes> using the <conjunctive_refinement_arcs>.
    Return True if a conjunctive refinement arc is detected between them, False otherwise.
    """
    if len(child_nodes) <= 1:
        return False

    x_min = min(
        min(child_node["bounding_rectangle"][:, 0]) for child_node in child_nodes
    )
    x_max = max(
        max(child_node["bounding_rectangle"][:, 0]) for child_node in child_nodes
    )
    y_min = min(parent_node["bounding_rectangle"][:, 1])
    y_max = min(
        max(child_node["bounding_rectangle"][:, 1]) for child_node in child_nodes
    )

    if y_max <= y_min or x_max <= x_min:
        return False

    for (x_center, y_center), _, _ in conjunctive_refinement_arcs:
        if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
            return True

    return False


def assign_refinement_to_node_objects(
    root_node: NodeDict, conjunctive_refinement_arcs: list[RotatedRectangle]
) -> NodeDict:
    """
    Assign refinement types OR(0)/AND(1) to all node objects in the tree rooted at the given <root_node> using the <conjunctive_refinement_arcs>. A parent node is assigned AND(1) if there is a conjunctive refinement arc between it and its child nodes, otherwise OR(0).
    Return the root node of the updated NodeDict tree.
    """
    queue: list[NodeDict] = [root_node]

    while queue:
        parent_node = queue.pop(0)
        child_nodes = parent_node["child_nodes"]
        if detect_conjunctive_refinement_arc(
            parent_node, child_nodes, conjunctive_refinement_arcs
        ):
            parent_node["refinement"] = 1
        else:
            parent_node["refinement"] = 0
        for child_node in child_nodes:
            queue.append(child_node)

    return root_node


def create_node_objects(
    image: np.ndarray,
    nodes: list[RotatedRectangle],
    refinement_edges: list[RotatedRectangle],
    countermeasure_edges: list[RotatedRectangle],
    conjunctive_refinement_arcs: list[RotatedRectangle],
) -> NodeDict | None:
    """
    Create node objects from the given <nodes> by using the <image>, <refinement_edges>, <countermeasure_edges>, and <conjunctive_refinement_arcs> to determine parent-child relationships, node types, and refinement types:
      1. assign the text labels for each node
      2. assign all child nodes for each parent node
      3. assign attack/defence types
      4. assign refinement types (OR/AND).
    Return the root node of the NodeDict tree or None if no nodes are correctly detected or no root node could be determined.
    """
    if len(nodes) == 0:
        print("Error: No nodes detected in the image.")
        return None

    nodes = nodes[::-1]

    node_objects_with_text = create_node_objects_with_text(image, nodes)

    root_node_with_child_nodes = assign_child_nodes_to_node_objects(
        node_objects_with_text, refinement_edges, countermeasure_edges
    )

    if root_node_with_child_nodes is None:
        print("Error: No root node could be determined.")
        return None

    root_node_with_type = assign_type_to_node_objects(root_node_with_child_nodes)

    root_node_with_refinement = assign_refinement_to_node_objects(
        root_node_with_type, conjunctive_refinement_arcs
    )

    return root_node_with_refinement


def prune_invalid_child_nodes(node: NodeDict) -> NodeDict:
    """
    For each <node>, if it has more than one countermeasure child node (switch_role == True), keep only the one with the most descendants. If that number is the same, keep the first one (leftmost).
    Return the pruned NodeDict tree root.
    """

    def count_descendants(node: NodeDict) -> int:
        return sum(
            1 + count_descendants(child_node) for child_node in node["child_nodes"]
        )

    pruned_child_nodes = [
        prune_invalid_child_nodes(child_node) for child_node in node["child_nodes"]
    ]
    countermeasure_indices = [
        index
        for index, child_node in enumerate(pruned_child_nodes)
        if child_node.get("switch_role", False)
    ]

    if len(countermeasure_indices) <= 1:
        node["child_nodes"] = pruned_child_nodes
        return node

    def score(index: int) -> tuple[int, float]:
        child_node = pruned_child_nodes[index]
        return (count_descendants(child_node), -index)

    best_countermeasure_child_node = max(countermeasure_indices, key=score)

    new_child_nodes: list[NodeDict] = []
    for index, child_node in enumerate(pruned_child_nodes):
        if not child_node.get("switch_role", False):
            new_child_nodes.append(child_node)
        elif index == best_countermeasure_child_node:
            new_child_nodes.append(child_node)

    node["child_nodes"] = new_child_nodes

    return node


def node_dict_tree_to_adt(root_node: NodeDict) -> ADT:
    """
    Convert a NodeDict tree into an ADTCreate.ADT tree recursively, starting from the given <root_node>.
    Return the ADT root object.
    """
    ADT.usedIDs.clear()

    adt_root = initADT(
        type=int(root_node["type"]),
        refinement=int(root_node["refinement"]),
        label=root_node["label"],
    )

    def add_child_nodes_recursive(parent_adt: ADT, parent_node: NodeDict) -> None:
        for child_node in parent_node["child_nodes"]:
            success, child_adt = parent_adt.addChild(
                typeChild=int(child_node["type"]),
                refinementChild=int(child_node["refinement"]),
                labelChild=child_node["label"],
                currentChild=parent_adt,
                tree=adt_root,
            )
            if not success or child_adt is None:
                print(
                    "Failed to add child node",
                    child_node["label"],
                    "to",
                    parent_node["label"],
                )
                continue
            add_child_nodes_recursive(child_adt, child_node)

    add_child_nodes_recursive(adt_root, root_node)

    return adt_root


def process_image(image: np.ndarray) -> ADT | None:
    """
    Process the given <image>:
      1. extract nodes
      2. extract connectors (refinement edges, countermeasure edges, conjunctive refinement arcs)
      3. create tree structure.
    Return the root ADT object or None if no nodes are correctly detected or no root node could be determined.
    """
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(grey, 200, 255, cv.THRESH_BINARY_INV)

    ### 1. Extract nodes: ###
    nodes = extract_nodes(threshold)

    ### 2. Extract connectors (refinement edges, countermeasure edges, conjunctive refinement arcs): ###
    # Build a mask of all nodes, slightly larger than the node rectangles (so connectors are not detected inside them)
    node_mask = np.zeros_like(threshold)
    for rectangle in nodes:
        center, (width, height), angle = rectangle
        expanded_rectangle = (
            center,
            (width * RECTANGLE_EXPANSION_FACTOR, height * RECTANGLE_EXPANSION_FACTOR),
            angle,
        )
        rectangle_points = cv.boxPoints(expanded_rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)
        cv.drawContours(node_mask, [rectangle_points], -1, (255, 0, 0), thickness=-1)

    # Remove node areas from threshold image
    masked_image = cv.bitwise_and(threshold, cv.bitwise_not(node_mask))

    connectors = extract_connectors(image, masked_image)

    refinement_edges, countermeasure_edges, conjunctive_refinement_arcs = (
        extract_edges_and_arcs(threshold, nodes, connectors)
    )

    ### 3. Create tree structure: ###
    node_dict_root = create_node_objects(
        image,
        nodes,
        refinement_edges,
        countermeasure_edges,
        conjunctive_refinement_arcs,
    )
    if node_dict_root is None:
        return None

    pruned_node_dict_root = prune_invalid_child_nodes(node_dict_root)
    return node_dict_tree_to_adt(pruned_node_dict_root)


def handle_single_image(image_path: str, output_path: str) -> None:
    """
    Process a single image at the given <image_path> and export the resulting ADT XML to <output_path>.xml.
    Return None (print status messages).
    """
    print(f"\nProcessing: {image_path}")
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return

    adt_root = process_image(image)
    if adt_root is not None:
        adt_root.exportXML(fileName=output_path, manual=False)
        print(f"ADT XML written to: {output_path}.xml")
    else:
        print(f"Failed to process image: {image_path}")


def main(input_path: str, output_directory_path: str) -> None:
    """
    The main entry point for processing images or directories of images located at the given <input_path>. Put the resulting XML files in the specified <output_directory_path>.
    Return None (write files and print status messages).
    """
    os.makedirs(output_directory_path, exist_ok=True)

    if os.path.isfile(input_path) and input_path.lower().endswith(".png"):
        output_file = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_directory_path, output_file)
        handle_single_image(input_path, output_path)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(".png"):
                input_file_path = os.path.join(input_path, filename)
                output_file = os.path.splitext(filename)[0]
                output_path = os.path.join(output_directory_path, output_file)
                handle_single_image(input_file_path, output_path)
    else:
        print(f"Error: {input_path} is not a .png file or directory.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_path> <output_directory_path>")
    else:
        main(sys.argv[1], sys.argv[2])
