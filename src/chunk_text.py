
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Find the project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Locate extracted text
INPUT_FILE = PROJECT_FOLDER / "output" / "extracted_requirements.txt"

# Output location
OUTPUT_FILE = PROJECT_FOLDER / "output" / "text_chunks.txt"

print("TESTGEN AI - TEXT CHUNKING")
print("--------------------------")

if not INPUT_FILE.exists():
    print("ERROR: Extracted text file not found!")
else:
    try:
        # Read extracted PDF text
        text = INPUT_FILE.read_text(encoding="utf-8")

        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Divide text into chunks
        chunks = splitter.split_text(text)

        print("Text loaded successfully!")
        print("Total chunks created:", len(chunks))

        # Save chunks
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            for index, chunk in enumerate(chunks, start=1):
                file.write(f"\n--- CHUNK {index} ---\n")
                file.write(chunk)
                file.write("\n")

        print("\nChunks saved successfully!")
        print("Output file:", OUTPUT_FILE)

        # Display first chunk
        if chunks:
            print("\nFIRST CHUNK:")
            print("--------------------------")
            print(chunks[0])

    except Exception as error:
        print("Chunking failed!")
        print("Error:", error)
