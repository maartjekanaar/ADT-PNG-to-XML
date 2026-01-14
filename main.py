# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Main file to process PNG images of Attack-Defence Trees and convert them into ADT XML files.

from ADTCreate.ADT import ADT
import cv2 as cv
import numpy as np
import os
import sys

from parameters import (
    RECTANGLE_EXPANSION_FACTOR,
    IMAGE_PREPROCESS_THRESHOLD,
)

from components_extraction import (
    extract_nodes,
    extract_connectors,
    extract_edges_and_arcs,
)

from tree_structure import (
    create_node_objects,
    prune_invalid_child_nodes,
    node_dict_tree_to_adt,
)

import analysis_functions
from analysis_functions import (
    show,
    show_all_components,
)


def process_image(image: np.ndarray) -> ADT | None:
    """
    Process the given <image>:
      1. extract nodes
      2. extract connectors (refinement edges, countermeasure edges, conjunctive refinement arcs)
      3. create tree structure.
    Return the root ADT object or None if no nodes are correctly detected or no root node could be determined.
    """
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(
        grey, IMAGE_PREPROCESS_THRESHOLD, 255, cv.THRESH_BINARY_INV
    )

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

    # For analysis:
    show(masked_image, "No nodes")

    connectors = extract_connectors(image, masked_image)

    refinement_edges, countermeasure_edges, conjunctive_refinement_arcs = (
        extract_edges_and_arcs(threshold, nodes, connectors)
    )

    # For analysis:
    show_all_components(
        image,
        nodes,
        refinement_edges,
        countermeasure_edges,
        conjunctive_refinement_arcs,
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

    # For analysis:
    analysis_functions.test_image = cv.imread(image_path)

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
