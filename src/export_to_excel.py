
from pathlib import Path
import re
import html
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# Find the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Input and output files
INPUT_FILE = PROJECT_FOLDER / "output" / "generated_test_cases.txt"
OUTPUT_FILE = PROJECT_FOLDER / "output" / "TestGen_AI_Test_Cases.xlsx"

print("TESTGEN AI - EXCEL EXPORT")
print("-------------------------")


def clean_cell(value):
    """Clean Markdown and HTML formatting from a table cell."""

    value = html.unescape(value)

    # Convert HTML line breaks into Excel line breaks
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)

    # Remove HTML tags
    value = re.sub(r"<[^>]+>", "", value)

    # Remove Markdown bold and italic markers
    value = value.replace("**", "")
    value = value.replace("*", "")

    # Remove extra spaces
    value = value.strip()

    return value


try:
    # Step 1: Check input file
    print("\nStep 1: Reading generated test cases...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "generated_test_cases.txt not found. Generate test cases first."
        )

    content = INPUT_FILE.read_text(encoding="utf-8")

    # Step 2: Find the test case table
    print("Step 2: Extracting test case table...")

    lines = content.splitlines()

    header = None
    test_cases = []

    for line in lines:

        line = line.strip()

        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]

        # Identify the table header
        if any("Test Case ID" in cell for cell in cells):
            header = cells
            continue

        # Ignore Markdown separator rows
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue

        # Read test case rows
        if header and len(cells) == len(header):

            cleaned_cells = [
                clean_cell(cell) for cell in cells
            ]

            if cleaned_cells[0].startswith("TC_"):
                test_cases.append(cleaned_cells)

    if not header or not test_cases:
        raise ValueError(
            "No test case table found in the generated output."
        )

    print("Total test cases extracted:", len(test_cases))

    # Step 3: Create DataFrame
    print("\nStep 3: Preparing Excel data...")

    cleaned_header = [
        clean_cell(column) for column in header
    ]

    dataframe = pd.DataFrame(
        test_cases,
        columns=cleaned_header
    )

    # Step 4: Export to Excel
    print("\nStep 4: Creating Excel file...")

    dataframe.to_excel(
        OUTPUT_FILE,
        index=False,
        sheet_name="Test Cases"
    )

    # Step 5: Format Excel workbook
    print("\nStep 5: Formatting Excel workbook...")

    workbook = load_workbook(OUTPUT_FILE)
    worksheet = workbook["Test Cases"]

    # Header formatting
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="17365D"
    )

    header_font = Font(
        bold=True,
        color="FFFFFF",
        size=11
    )

    thin_border = Border(
        bottom=Side(style="thin", color="D9E2F3")
    )

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

    # Format data cells
    for row in worksheet.iter_rows(min_row=2):

        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )
            cell.border = thin_border

    # Set column widths
    column_widths = {
        "Test Case ID": 18,
        "Test Scenario": 35,
        "Test Type": 22,
        "Preconditions": 40,
        "Test Steps": 55,
        "Test Data": 40,
        "Expected Result": 55,
        "Priority": 15
    }

    for column_number, column_name in enumerate(
        cleaned_header, start=1
    ):
        letter = get_column_letter(column_number)

        worksheet.column_dimensions[letter].width = (
            column_widths.get(column_name, 25)
        )

    worksheet.row_dimensions[1].height = 35

    # Adjust row heights based on content
    for row_number in range(2, worksheet.max_row + 1):
        worksheet.row_dimensions[row_number].height = 100

    # Freeze header and enable filters
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # Save formatting
    workbook.save(OUTPUT_FILE)

    # Final result
    print("\n--------------------------------")
    print("EXCEL EXPORT COMPLETED!")
    print("--------------------------------")

    print("Total test cases:", len(test_cases))
    print("Excel file saved at:")
    print(OUTPUT_FILE)

except Exception as error:
    print("\nERROR: Excel export failed!")
    print(error)
