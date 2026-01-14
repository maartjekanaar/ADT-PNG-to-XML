# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Parameters for image processing and analysis.

# General parameters:
RECTANGLE_EXPANSION_FACTOR = 1.25

# Helper functions parameters:
ERODE_PIXELS = 4
POINT_NEAR_RECTANGLE_THRESHOLD = 15

# Components extraction parameters:
IOU_THRESHOLD = 0.3
NODES_KERNEL_SIZE = (2, 2)  # for dilating during node extraction
NODES_MIN_CONTOUR_AREA = 200
NODES_MIN_WIDTH = 40
NODES_MIN_HEIGHT = 20
NODES_MIN_RECTANGLE_AREA = 2000
NODES_MAX_RECTANGLE_AREA = 100000
NODES_MIN_INK_DENSITY = 0.01
CONNECTORS_ORIENTATION_BIN_SIZE = 15
CONNECTORS_MERGE_THRESHOLD = 250
CONNECTORS_KERNEL_SIZE = (2, 2)  # for dilating during connector extraction
CONNECTORS_BLUR_KERNEL_SIZE = 3  # for median blur during connector extraction
CONNECTORS_THRESHOLD = 12
CONNECTORS_MIN_LINE_LENGTH = 10
CONNECTORS_MAX_LINE_GAP = 40
EDGES_CONTOUR_THRESHOLD = 4

# Tree structure parameters:
ROI_THRESHOLD = 200

# Main file parameters:
IMAGE_PREPROCESS_THRESHOLD = 200
