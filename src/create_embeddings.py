
from pathlib import Path
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Find the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Input file
INPUT_FILE = PROJECT_FOLDER / "output" / "extracted_requirements.txt"

# Output folder
OUTPUT_FOLDER = PROJECT_FOLDER / "output"

# Files to save
INDEX_FILE = OUTPUT_FOLDER / "faiss_index.bin"
CHUNKS_FILE = OUTPUT_FOLDER / "chunks.json"

print("TESTGEN AI - CREATING EMBEDDINGS")
print("--------------------------------")

if not INPUT_FILE.exists():
    print("ERROR: Extracted requirements file not found!")

else:
    try:
        # Step 1: Read extracted text
        print("\nStep 1: Reading requirements...")

        text = INPUT_FILE.read_text(encoding="utf-8")

        print("Requirements loaded successfully!")

        # Step 2: Split text into chunks
        print("\nStep 2: Creating text chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        print("Total chunks:", len(chunks))

        # Step 3: Load the embedding model
        print("\nStep 3: Loading the AI embedding model...")
        print("The first download may take a few minutes.")

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print("Embedding model loaded successfully!")

        # Step 4: Convert chunks into embeddings
        print("\nStep 4: Converting text into embeddings...")

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        print("Embeddings created successfully!")
        print("Embedding shape:", embeddings.shape)

        # Step 5: Create FAISS index
        print("\nStep 5: Creating FAISS index...")

        index = faiss.IndexFlatIP(embeddings.shape[1])

        index.add(embeddings)

        print("FAISS index created successfully!")

        # Step 6: Save FAISS index
        print("\nStep 6: Saving files...")

        OUTPUT_FOLDER.mkdir(exist_ok=True)

        faiss.write_index(index, str(INDEX_FILE))

        # Save original chunks separately
        with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=4)

        print("\n--------------------------------")
        print("EMBEDDINGS CREATED SUCCESSFULLY!")
        print("--------------------------------")

        print("Total chunks:", len(chunks))
        print("Embedding dimensions:", embeddings.shape[1])
        print("FAISS index:", INDEX_FILE)
        print("Chunks file:", CHUNKS_FILE)

    except Exception as error:
        print("\nERROR: Embedding creation failed!")
        print(error)
