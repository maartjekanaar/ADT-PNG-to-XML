# Developed by Maartje Kanaar as part of the Bachelor Thesis Project for the Leiden Institute of Advanced Sciences (LIACS).
# Assessment file to compare XML files of Attack-Defence Trees.

from typing import Optional
from ADTCreate.ADT import ADT
import os
import sys


def handle_single_comparison(
    generated_xml_path: str, reference_xml_path: str
) -> Optional[tuple[bool, int | None]]:
    """
    Compare a single ADT at the given <generated_xml_path> to the ADT at the <reference_xml_path>.
    Return a tuple containing a boolean indicating equivalence and an integer representing the Levenshtein distance or None if not equivalent, or None if an error occurs.
    """
    print(f"\nProcessing: {generated_xml_path}")
    generated_adt = ADT(0, 0, 0, "", None, [(True, None)])
    reference_adt = ADT(0, 0, 0, "", None, [(True, None)])
    generated_adt_root, _ = generated_adt.importXML(
        generated_xml_path[:-4], manual=False
    )
    reference_adt_root, _ = reference_adt.importXML(
        reference_xml_path[:-4], manual=False
    )

    if not isinstance(generated_adt_root, ADT) or not isinstance(
        reference_adt_root, ADT
    ):
        print("Error importing one of the XML files.")
        return

    generated_adterm = generated_adt_root.generateADTerm()
    reference_adterm = reference_adt_root.generateADTerm()

    comparison_result = generated_adt_root.compareADTerms(
        generated_adterm, reference_adterm
    )

    return comparison_result


def main(generated_path: str, reference_path: str) -> None:
    """
    The main entry point for comparing directories of ADT XML files located at the given <generated_path> and <reference_path>.
    Return None (write files and print status messages).
    """
    if (
        os.path.isfile(generated_path)
        and generated_path.lower().endswith(".xml")
        and os.path.isfile(reference_path)
        and reference_path.lower().endswith(".xml")
    ):
        comparison_result = handle_single_comparison(generated_path, reference_path)
        if comparison_result:
            equivalent, distance = comparison_result
            print(
                f"Single file comparison result: Equivalent: {equivalent}, Distance: {distance}"
            )

    elif os.path.isdir(generated_path) and os.path.isdir(reference_path):
        results = []
        for filename in os.listdir(generated_path):
            if filename.lower().endswith(".xml"):
                generated_xml_path = os.path.join(generated_path, filename)
                reference_xml_path = os.path.join(reference_path, filename)
                if os.path.exists(reference_xml_path):
                    comparison_result = handle_single_comparison(
                        generated_xml_path, reference_xml_path
                    )
                    if comparison_result:
                        equivalent, distance = comparison_result
                        results.append(f"{filename}: {equivalent}, {distance}")
                else:
                    print(f"No matching file for {filename} in {reference_path}.")

        if results:
            summary_path = os.path.join(generated_path, "comparison_results.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                for line in results:
                    f.write(line + "\n")
    else:
        print(
            f"Error: {generated_path} and/or {reference_path} is not an .xml file or a directory."
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <generated_path> <reference_path>")
    else:
        main(sys.argv[1], sys.argv[2])
