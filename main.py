from ADTCreate.ADT import ADT, initADT
from typing import Literal, TypedDict
import cv2 as cv
import math
import numpy as np
import os
import pytesseract
import re
import sys
import xml.etree.ElementTree as ET


### Type aliases and data structures: ###


Point = tuple[int, int]
Line = tuple[Point, Point]
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


def draw_lines(image: np.ndarray, lines: list[Line]) -> np.ndarray:
    """
    Draw the given <lines> in blue on a blank white image of the same shape as the input <image>.
    Return the image with the lines drawn.
    """
    out = np.ones_like(image) * 255  # blank white
    for p1, p2 in lines:
        cv.line(out, p1, p2, (255, 0, 0), 2)  # blue

    return out


def point_near_or_in_rectangle(
    point: tuple[float, float], rectangle_points: RectanglePoints, threshold: int = 5
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


def _xml_node_label(xml_node: ET.Element) -> str:
    """
    Internal function.
    Extract and return the label text from the given <xml_node> element.
    """
    label_element = xml_node.find("label")
    return (label_element.text or "").strip() if label_element is not None else ""


def _patch_switch_role_recursive(
    xml_node: ET.Element, node_dict_node: NodeDict
) -> bool:
    """
    Internal function.
    Update the 'switchRole' attribute for the given <xml_node> to match the <node_dict_node>. Recursively process child nodes.
    Return True if successful, False if structure mismatch.
    """
    switch_role_value = node_dict_node["switch_role"]

    if switch_role_value:
        xml_node.set("switchRole", "yes")
    elif "switchRole" in xml_node.attrib:
        del xml_node.attrib["switchRole"]

    xml_child_nodes = sorted(xml_node.findall("node"), key=_xml_node_label)
    node_dict_child_nodes = sorted(
        node_dict_node["child_nodes"], key=lambda node: node["label"]
    )

    if len(xml_child_nodes) != len(node_dict_child_nodes):
        print(
            f"Error: XML has {len(xml_child_nodes)} child nodes, NodeDict has {len(node_dict_child_nodes)} child nodes."
        )
        return False

    for xml_child_node, node_dict_child_node in zip(
        xml_child_nodes, node_dict_child_nodes
    ):
        if not _patch_switch_role_recursive(xml_child_node, node_dict_child_node):
            return False

    return True


### Main processing functions: ###


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

    for i in range(len(rectangles)):
        if used_rectangles[i]:
            continue
        x_left1, y_top1, width1, height1 = bounding_rectangles[i]
        rectangle1 = (x_left1, y_top1, x_left1 + width1, y_top1 + height1)

        # Compare with other remaining rectangles
        for j in range(i + 1, len(rectangles)):
            if used_rectangles[j]:
                continue
            x_left2, y_top2, width2, height2 = bounding_rectangles[j]
            rectangle2 = (x_left2, y_top2, x_left2 + width2, y_top2 + height2)

            xx1 = max(rectangle1[0], rectangle2[0])
            yy1 = max(rectangle1[1], rectangle2[1])
            xx2 = min(rectangle1[2], rectangle2[2])
            yy2 = min(rectangle1[3], rectangle2[3])
            intersection = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area1 = (rectangle1[2] - rectangle1[0]) * (rectangle1[3] - rectangle1[1])
            area2 = (rectangle2[2] - rectangle2[0]) * (rectangle2[3] - rectangle2[1])
            union = area1 + area2 - intersection

            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                used_rectangles[j] = True  # mark as duplicate

        used_rectangles[i] = True
        merged_rectangles.append(rectangles[i])

    return merged_rectangles


def extract_nodes(threshold: np.ndarray) -> list[RotatedRectangle]:
    """
    Extract node regions (ellipses and rectangles) from the given <threshold> binary image.
    Return a list of the detected node rectangles.
    """
    raw_nodes: list[RotatedRectangle] = []

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    clean = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(clean, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)

        # Skip very small contours (noise, small text)
        if area < 200:
            continue

        rectangle = cv.minAreaRect(contour)
        _, (width, height), _ = rectangle

        # Skip narrow contours (likely not nodes)
        if width < 40 or height < 20:
            continue

        # Skip small (other shapes) or large (merged groups) rectangles
        if (width * height) < 2000 or (width * height) > 100000:
            continue

        # Only keep rectangles with moderate aspect ratios
        aspect_ratio = max(width, height) / min(width, height)
        if 1.5 < aspect_ratio < 5.0:
            raw_nodes.append(rectangle)

    node_rectangles = merge_overlapping_rectangles(raw_nodes)

    return node_rectangles


def merge_overlapping_lines(
    image: np.ndarray, lines: list[Line]
) -> list[RotatedRectangle]:
    """
    Merge overlapping given <lines> into rotated rectangles by drawing all lines in the given <image> and finding contours around them.
    Return a list of the rotated rectangles representing the merged lines.
    """
    merged_lines: list[RotatedRectangle] = []

    lines_image = draw_lines(image, lines)
    grey = cv.cvtColor(lines_image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(grey, 250, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rectangle = cv.minAreaRect(contour)
        (x_center, y_center), (width, height), angle = rectangle
        merged_lines.append(
            (
                (x_center, y_center),
                (
                    width * RECTANGLE_EXPANSION_FACTOR,
                    height * RECTANGLE_EXPANSION_FACTOR,
                ),
                angle,
            )
        )

    return merged_lines


def extract_refinement_edges(
    image: np.ndarray, masked_image: np.ndarray
) -> tuple[list[RotatedRectangle], list[Line]]:
    """
    Extract refinement edges (continuous lines) from the given <masked_image> using probabilistic hough transform. Let merge_overlapping_lines() transform them into rotated rectangles on the given <image>.
    Return a tuple of the rotated rectangles for the refinement edges and the raw line segments for masking.
    """
    raw_refinement_edges: list[Line] = []

    solid_input = cv.medianBlur(masked_image, 3)
    lines = cv.HoughLinesP(
        solid_input,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=5,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            delta_x = x2 - x1
            delta_y = y2 - y1
            length = math.hypot(delta_x, delta_y)

            # Skip short segments (likely not refinement edges)
            if length < 40:
                continue

            # Skip (almost) horizontal segments (conjunctive refinement arcs)
            vertical_fraction = abs(delta_y) / (length + 1e-6)
            if vertical_fraction < 0.3:
                continue

            raw_refinement_edges.append(((x1, y1), (x2, y2)))

    refinement_edges = merge_overlapping_lines(image, raw_refinement_edges)

    return refinement_edges, raw_refinement_edges


def extract_conjunctive_refinement_arcs(
    masked_image: np.ndarray,
) -> list[RotatedRectangle]:
    """
    Extract conjunctive refinement arcs (curved horizontal lines) from the given <masked_image>.
    Return a list of the rotated rectangles for the refinement arcs.
    """
    conjunctive_refinement_arcs: list[RotatedRectangle] = []

    contours, _ = cv.findContours(masked_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv.contourArea(contour)

        # Skip small contours (noise, small text)
        if area < 20:
            continue

        rectangle = cv.minAreaRect(contour)
        (x_center, y_center), (width, height), angle = rectangle

        # Skip too small or too long contours
        if max(width, height) < 20 or min(width, height) > 40:
            continue

        # Skip too vertical contours
        diagonal = math.hypot(width, height)
        vertical_fraction = min(width, height) / (diagonal + 1e-6)
        if vertical_fraction > 0.5:
            continue

        conjunctive_refinement_arcs.append(
            (
                (x_center, y_center),
                (
                    width * RECTANGLE_EXPANSION_FACTOR,
                    height * RECTANGLE_EXPANSION_FACTOR,
                ),
                angle,
            )
        )

    return conjunctive_refinement_arcs


def extract_countermeasure_edges(
    image: np.ndarray, masked_image: np.ndarray
) -> list[RotatedRectangle]:
    """
    Extract countermeasure edges (broken lines) from the given <masked_image> using probabilistic Hough transform. Let merge_overlapping_lines() transform them into rotated rectangles on the given <image>.
    Return a list of the rotated rectangles for the countermeasure edges.
    """
    raw_countermeasure_edges: list[Line] = []

    solid_input = cv.medianBlur(masked_image, 3)
    lines = cv.HoughLinesP(
        solid_input,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=4,
        maxLineGap=100,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            raw_countermeasure_edges.append(((x1, y1), (x2, y2)))

    countermeasure_edges = merge_overlapping_lines(image, raw_countermeasure_edges)

    return countermeasure_edges


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
        _, roi_threshold = cv.threshold(roi_grey, 50, 255, cv.THRESH_BINARY)

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
    parent: NodeDict, child_node: NodeDict, edges: list[RotatedRectangle]
) -> bool:
    """
    Detect whether there is an edge (refinement or countermeasure) between the given <parent> node and <child_node> using the list of <edges>.
    Return True if an edge exists, False otherwise.
    """
    for edge in edges:
        rectangle_points = cv.boxPoints(edge)
        rectangle_points = np.round(rectangle_points).astype(np.intp)

        y_coordinates = rectangle_points[:, 1]
        index_high = np.argmin(y_coordinates)
        index_low = np.argmax(y_coordinates)
        point_high = tuple(rectangle_points[index_high])
        point_low = tuple(rectangle_points[index_low])

        if point_near_or_in_rectangle(
            point_high, parent["bounding_rectangle"]
        ) and point_near_or_in_rectangle(point_low, child_node["bounding_rectangle"]):
            return True

    return False


def assign_child_nodes_to_node_objects(
    node_objects: list[NodeDict],
    refinement_edges: list[RotatedRectangle],
    countermeasure_edges: list[RotatedRectangle],
) -> NodeDict:
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
        parent_node["child_nodes"] = child_nodes

    root_node: NodeDict | None = None
    for node in unassigned_child_nodes:
        if node["child_nodes"] != []:
            root_node = node
            break
    if root_node is None:
        # Select the highest node vertically (the node with the lowest y-coordinate) as root_node
        root_node = min(
            node_objects, key=lambda node: node["bounding_rectangle"][:, 1].min()
        )

    updated_root_node = next(
        node for node in node_objects if is_same_node(node, root_node)
    )

    return updated_root_node


def assign_type_to_node_objects(root_node: NodeDict) -> NodeDict:
    """
    Assign attack(0)/defence(1) types to all node objects in the tree rooted at the given <root_node>, based on 'switch_role' attributes. The root node is assigned type 0 (attack) and types propagate down the tree.
    Return the root node of the updated NodeDict tree.
    """
    root_node["type"] = 0  # root is attack
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
    parent: NodeDict,
    child_nodes: list[NodeDict],
    conjunctive_refinement_arcs: list[RotatedRectangle],
) -> bool:
    """
    Detect whether there is a conjunctive refinement arc in the rectangular area between the given <parent> node and <child_nodes> using the <conjunctive_refinement_arcs>.
    Return True if a conjunctive refinement arc is detected, False otherwise.
    """
    if len(child_nodes) <= 1:
        return False

    x_min = min(
        min(child_node["bounding_rectangle"][:, 0]) for child_node in child_nodes
    )
    x_max = max(
        max(child_node["bounding_rectangle"][:, 0]) for child_node in child_nodes
    )
    y_min = min(parent["bounding_rectangle"][:, 1])
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
            parent_node["refinement"] = 1  # AND
        else:
            parent_node["refinement"] = 0  # OR
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
    Return the root node of the NodeDict tree or None if no nodes are (correctly) detected.
    """
    if len(nodes) == 0:
        print(f"Error: No nodes detected in the image.")
        return None

    nodes = nodes[::-1]

    node_objects_with_text = create_node_objects_with_text(image, nodes)

    root_node_with_child_nodes = assign_child_nodes_to_node_objects(
        node_objects_with_text, refinement_edges, countermeasure_edges
    )

    root_node_with_type = assign_type_to_node_objects(root_node_with_child_nodes)

    root_node_with_refinement = assign_refinement_to_node_objects(
        root_node_with_type, conjunctive_refinement_arcs
    )

    return root_node_with_refinement


def node_dict_tree_to_adt(root_node: NodeDict) -> ADT:
    """
    Convert a NodeDict tree (with child_nodes) into an ADTCreate.ADT tree recursively, starting from the given <root_node>.
    Return the ADT root object.
    """
    ADT.usedIDs.clear()

    adt_root = initADT(
        type=int(root_node["type"]),
        refinement=int(root_node["refinement"]),
        label=root_node["label"],
    )

    def add_child_nodes_recursive(parent_adt: ADT, parent_node: NodeDict) -> None:
        for child in parent_node["child_nodes"]:
            success, child_adt = parent_adt.addChild(
                typeChild=int(child["type"]),
                refinementChild=int(child["refinement"]),
                labelChild=child["label"],
                currentChild=parent_adt,
                tree=adt_root,
            )
            if not success or child_adt is None:
                print("Failed to add child", child["label"], "to", parent_node["label"])
                continue
            add_child_nodes_recursive(child_adt, child)

    add_child_nodes_recursive(adt_root, root_node)

    return adt_root


def process_image(image: np.ndarray) -> tuple[ADT, NodeDict] | None:
    """
    Process the given <image>:
      1. extract nodes (ellipses + rectangles)
      2. extract refinement edges (continuous lines)
      3. extract conjunctive refinement arcs (curved horizontal lines)
      4. extract countermeasure edges (broken lines)
      5. create tree structure.
    Return the root ADT object and NodeDict root or None if no nodes are (correctly) detected.
    """
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(grey, 200, 255, cv.THRESH_BINARY_INV)

    ### 1. Extract nodes (ellipses + rectangles): ###
    nodes = extract_nodes(threshold)

    ### 2. Extract refinement edges (continuous lines): ###
    # Build a mask of all nodes, slightly larger than the node rectangles (so edges are not detected inside them)
    node_mask = np.zeros_like(threshold)
    for rectangle in nodes:
        (x_center, y_center), (width, height), angle = rectangle
        expanded_rectangle = (
            (x_center, y_center),
            (width * RECTANGLE_EXPANSION_FACTOR, height * RECTANGLE_EXPANSION_FACTOR),
            angle,
        )
        rectangle_points = cv.boxPoints(expanded_rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)
        cv.drawContours(node_mask, [rectangle_points], -1, (255, 0, 0), thickness=-1)

    # Remove node areas from threshold image
    masked_image = cv.bitwise_and(threshold, cv.bitwise_not(node_mask))

    refinement_edges, raw_refinement_edges = extract_refinement_edges(
        image, masked_image
    )

    ### 3. Extract conjunctive refinement arcs (curved horizontal lines): ###
    # Build an extended mask of all refinement_edges
    added_refinement_edge_mask = np.zeros_like(masked_image)
    for p1, p2 in raw_refinement_edges:
        cv.line(added_refinement_edge_mask, p1, p2, (255, 0, 0), thickness=5)

    # Remove refinement edge areas from masked_image
    second_masked_image = cv.bitwise_and(
        masked_image, cv.bitwise_not(added_refinement_edge_mask)
    )

    conjunctive_refinement_arcs = extract_conjunctive_refinement_arcs(
        second_masked_image
    )

    ### 4. Extract countermeasure edges (broken lines): ###
    # Build an extended mask of all conjunctive_refinement_arcs
    added_conjunctive_refinement_arc_mask = np.zeros_like(second_masked_image)
    for rectangle in conjunctive_refinement_arcs:
        rectangle_points = cv.boxPoints(rectangle)
        rectangle_points = np.round(rectangle_points).astype(np.intp)
        cv.drawContours(
            added_conjunctive_refinement_arc_mask,
            [rectangle_points],
            -1,
            (255, 0, 0),
            thickness=-1,
        )

    # Remove conjunctive refinement arc areas from second_masked_image
    third_masked_image = cv.bitwise_and(
        second_masked_image, cv.bitwise_not(added_conjunctive_refinement_arc_mask)
    )

    countermeasure_edges = extract_countermeasure_edges(image, third_masked_image)

    ### 5. Create tree structure: ###
    node_dict_root = create_node_objects(
        image,
        nodes,
        refinement_edges,
        countermeasure_edges,
        conjunctive_refinement_arcs,
    )
    if node_dict_root is None:
        return None

    adt_root = node_dict_tree_to_adt(node_dict_root)

    return adt_root, node_dict_root


def export_adt_to_xml(
    adt_root: ADT, node_dict_root: NodeDict, output_path: str
) -> bool:
    """
    Export the ADT tree rooted at the <adt_root> to XML at <output_path>.xml, patching the switchRole attributes using the given NodeDict tree rooted at <node_dict_root>.
    Return True if successful, False otherwise.
    """
    adt_root.exportXML(fileName=output_path, manual=False)
    xml_path = output_path + ".xml"
    xml_tree = ET.parse(xml_path)
    xml_root_element = xml_tree.getroot()
    xml_root_node = xml_root_element.find("node")

    if xml_root_node is None or not _patch_switch_role_recursive(
        xml_root_node, node_dict_root
    ):
        return False

    xml_tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return True


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

    roots = process_image(image)
    if roots is not None:
        adt_root, node_dict_root = roots
        if export_adt_to_xml(adt_root, node_dict_root, output_path):
            print(f"ADT XML written to: {output_path}.xml")
        else:
            print(
                f"Failed to patch the switch role in the ADT XML for image: {image_path}"
            )
            adt_root.exportXML(fileName=output_path, manual=False)
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
