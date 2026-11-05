
from pathlib import Path
from pypdf import PdfReader

# Get the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Locate the PDF
PDF_PATH = PROJECT_FOLDER / "documents" / "skill_barter_requirements.pdf"

print("TESTGEN AI - PDF TEXT EXTRACTION")
print("--------------------------------")

# Check whether PDF exists
if not PDF_PATH.exists():
    print("ERROR: PDF file not found!")
    print("Please check your documents folder.")
else:
    try:
        # Open the PDF
        reader = PdfReader(str(PDF_PATH))

        print("PDF opened successfully!")
        print("Total pages:", len(reader.pages))

        # Extract text from every page
        extracted_text = ""

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            extracted_text += (
                f"\n--- PAGE {page_number} ---\n"
                + page_text
            )

        # Display extracted text
        print("\nPDF TEXT EXTRACTED SUCCESSFULLY!")
        print("--------------------------------")

        print(extracted_text)

        # Save extracted text
        output_folder = PROJECT_FOLDER / "output"
        output_folder.mkdir(exist_ok=True)

        output_file = output_folder / "extracted_requirements.txt"
        output_file.write_text(extracted_text, encoding="utf-8")

        print("\n--------------------------------")
        print("Text saved successfully!")
        print("File location:", output_file)

    except Exception as error:
        print("PDF extraction failed!")
        print("Error:", error)
