# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Functions for creating the tree structure.

from ADTCreate.ADT import ADT, initADT
import cv2 as cv
import numpy as np
import pytesseract
import re

from type_aliases_and_data_structures import (
    RotatedRectangle,
    NodeDict,
)

from helper_functions import (
    get_line_endpoints,
    point_near_or_in_rectangle,
    is_same_node,
)


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
