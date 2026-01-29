# Bachelor Thesis Project: ADT Image to XML Converter

This repository contains the Python program for Maartje Kanaar's Bachelor Thesis Project at the Leiden Institute of Advanced Sciences (LIACS).

## Project Overview

This tool converts **Attack-Defence Tree (ADT)** diagrams in PNG format into XML files that conform to a provided schema. ADTs are graphical models used in security analysis to represent possible attack scenarios and corresponding defence strategies.

The script can process either a single PNG file or a directory containing multiple PNG files. For each image, it extracts the relevant information and generates a corresponding XML file, storing the results in the specified output directory.

## Features

- Converts ADT diagrams from PNG images to XML format
- Supports single files and batch processing of folders
- Automatically creates the output directory if it does not exist
- Compares generated XML files with reference XML files, reporting equivalence and Levenshtein distance

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [File Overview](#file-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Note on ADT Create Library Modification](#note-on-adt-create-library-modification)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## File Overview

This section provides a brief overview of the main files in this repository:

- **main.py**: Main entry point for converting PNG images of ADT diagrams to XML files. Handles both single images and batch processing of directories.
- **type_aliases_and_data_structures.py**: Contains type aliases and data structures used for type hinting and organisation.
- **parameters.py**: Stores configuration constants and parameters used for image processing and extraction.
- **helper_functions.py**: Helper utilities used throughout the codebase for various supporting tasks.
- **components_extraction.py**: Contains functions for extracting nodes, connectors, and edges/arcs from preprocessed images.
- **tree_structure.py**: Functions for building the tree structure from extracted components, pruning invalid nodes, and converting to ADT objects.
- **analysis_functions.py**: Utility functions for visualising and analysing intermediate image processing results (e.g., displaying images with detected components).
- **compare.py**: Script to compare generated ADT XML files with reference XML files. Supports both single file and directory comparison, and outputs equivalence and Levenshtein distance.
- **ADTCreate/**: Contains a modified version of the ADT Create library, used for ADT object creation, XML import/export, and comparison.

## Prerequisites

Ensure the following are installed on your machine:

- **Python 3.12** (or higher)
- **pip** (Python package manager)
- **Tesseract-OCR** (text recognition engine)

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/maartjekanaar/ADT-PNG-to-XML.git
   cd path/to/ADT-PNG-to-XML
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On Windows:

     ```bash
     .\.venv\Scripts\activate
     ```

   - On Linux/macOS:

     ```bash
     source .venv/bin/activate
     ```

4. **Upgrade pip (recommended):**

   ```bash
   pip install --upgrade pip
   ```

5. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the dependencies, you can use the following scripts:

### 1. Convert ADT PNG Images to XML

Run the main conversion script:

```bash
python main.py <input_path> <output_directory_path>
```

- `<input_path>`: Path to either a single PNG file of an ADT image **or** a directory containing one or more PNG files of ADT images.
- `<output_directory_path>`: Path to a directory where the resulting XML file(s) will be stored. The directory will be created if it does not exist.

**Examples:**

Convert a single PNG image:

```bash
python main.py ./path/to/ADT.png ./path/to/folder-for-XML/
```

Convert all PNG images in a folder:

```bash
python main.py ./path/to/ADT-images/ ./path/to/folder-for-XML/
```

The script will process the input PNG image(s) and generate XML file(s) in the specified output directory. If the output directory does not exist, it will be created automatically.

**Note for Fast Conversion:**

For fast conversion, **comment out all lines directly after `For analysis:` in the code**. These lines trigger the display of intermediate results as images and are only needed for debugging, experiments or analysis. Commenting them out will prevent pop-up windows and speed up the conversion process.

### 2. Compare Generated XML Files

You can compare generated XML files with reference XML files using:

```bash
python compare.py <generated_path> <reference_path>
```

- `<generated_path>`: Path to a single generated XML file **or** a directory containing generated XML files.
- `<reference_path>`: Path to a single reference XML file **or** a directory containing reference XML files (with matching filenames).

**Examples:**

Compare two single XML files:

```bash
python compare.py ./generated/ADT.xml ./reference/ADT.xml
```

Compare all XML files in two directories:

```bash
python compare.py ./generated-xml/ ./reference-xml/
```

The script will print the equivalence and Levenshtein distance for each comparison. For directory comparisons, a summary file `comparison_results.txt` will be written to the generated directory.

## Troubleshooting

- **Tesseract-OCR not found:** Ensure Tesseract is installed and added to your system PATH. On Windows, you may need to specify the Tesseract executable path in your script or environment variables.
- **Permission errors:** Make sure you have write access to the output directory.
- **Invalid input:** Check that your input path points to valid PNG files or a directory containing PNG files.

## Note on ADT Create Library Modification

This repository includes a modified copy of the ADT Create library. The modification was necessary to address an implementation issue with the assignment and interpretation of the `switchRole` attribute in both the XML export and import functionalities. In the original ADT Create library:

- In `exportXML()`, the `switchRole` attribute was set to "yes" for all defence nodes, regardless of whether their role actually differed from that of their parent node.
- In `importXML()`, all nodes with `switchRole` set to "yes" were interpreted as defence nodes, regardless of their actual intended role.

To ensure correct XML output and input, the ADT Create library was copied into this project and updated so that the `switchRole` attribute is only set to "yes" for nodes whose role differs from their parent node when exporting, and is only interpreted as a role switch when appropriate during import. The modified version is included in this repository.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/maartjekanaar/ADT-PNG-to-XML.git) or contact Maartje Kanaar via the repository profile.
