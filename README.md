
# Bachelor Thesis Project: ADT Image to XML Converter

This repository contains the Python program for Maartje Kanaar's Bachelor Thesis Project at the Leiden Institute of Advanced Sciences (LIACS).

## Project Overview

This tool converts **Attack-Defence Tree (ADT)** diagrams in PNG format into XML files that conform to a provided schema. ADTs are graphical models used in security analysis to represent possible attack scenarios and corresponding defence strategies.

The script can process either a single PNG file or a directory containing multiple PNG files. For each image, it extracts the relevant information and generates a corresponding XML file, storing the results in the specified output directory.

## Features

- Converts ADT diagrams from PNG images to XML format
- Supports single files and batch processing of folders
- Automatically creates the output directory if it does not exist
- Validates output against the ADT XML schema

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

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

After installing the dependencies, you can convert ADT images in PNG format to XML files by running:

```bash
python main.py <input_path> <output_directory_path>
```

- `<input_path>`: Path to either a single PNG file of an ADT image **or** a directory containing one or more PNG files of ADT images.
- `<output_directory_path>`: Path to a directory where the resulting XML file(s) will be stored. The directory will be created if it does not exist.

### Examples

Convert a single PNG image:

```bash
python main.py ./path/to/ADT.png ./path/to/folder-for-XML/
```

Convert all PNG images in a folder:

```bash
python main.py ./path/to/ADT-images/ ./path/to/folder-for-XML/
```

The script will process the input PNG image(s) and generate XML file(s) in the specified output directory. If the output directory does not exist, it will be created automatically.

## Troubleshooting

- **Tesseract-OCR not found:** Ensure Tesseract is installed and added to your system PATH. On Windows, you may need to specify the Tesseract executable path in your script or environment variables.
- **Permission errors:** Make sure you have write access to the output directory.
- **Invalid input:** Check that your input path points to valid PNG files or a directory containing PNG files.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/maartjekanaar/ADT-PNG-to-XML.git) or contact Maartje Kanaar via the repository profile.
