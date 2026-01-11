# Bachelor Thesis Project

This repository contains the Python program for Maartje Kanaar's Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).

This program may be used to convert Attack-Defence Trees (ADTs) in PNG format to XML format.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Ensure the following is installed on your machine:

- **Python 3.12** (or higher)
- **pip** (Python package manager)
- **Tesseract-OCR** (text recognition engine)

## Installation

Follow these steps to set up the project:

1. Clone the repository:

   ```bash
   git clone <https://github.com/maartjekanaar/ADT-PNG-to-XML.git>
   cd path/to/ADT-PNG-to-XML
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   On Windows, run:

   ```bash
   .\.venv\Scripts\activate
   ```

   On Linux/macOS, run:

   ```bash
   source .venv/bin/activate
   ```

4. Upgrade pip (recommended):

   ```bash
   pip install --upgrade pip
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the dependencies, you can run the program by executing:

```bash
python main.py <input_path> <output_directory_path>
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
