
from pathlib import Path
import json
import faiss

from sentence_transformers import SentenceTransformer


# Find project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# File locations
INDEX_FILE = PROJECT_FOLDER / "output" / "faiss_index.bin"
CHUNKS_FILE = PROJECT_FOLDER / "output" / "chunks.json"

print("TESTGEN AI - REQUIREMENT RETRIEVAL")
print("----------------------------------")


try:
    # Step 1: Check whether required files exist
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "FAISS index or chunks file is missing. Run create_embeddings.py first."
        )

    # Step 2: Load FAISS index
    print("\nStep 1: Loading FAISS index...")

    index = faiss.read_index(str(INDEX_FILE))

    # Step 3: Load original text chunks
    print("Step 2: Loading requirement chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Total chunks loaded:", len(chunks))

    # Step 4: Load embedding model
    print("\nStep 3: Loading embedding model...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!")

    # Step 5: Get user's question
    print("\n----------------------------------")

    question = input(
        "\nEnter your software testing question: "
    ).strip()

    if not question:
        raise ValueError("Please enter a question.")

    # Step 6: Convert question into embedding
    print("\nStep 4: Converting question into embedding...")

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype("float32")

    # Step 7: Search FAISS
    print("\nStep 5: Searching relevant requirements...")

    number_of_results = min(3, len(chunks))

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    # Step 8: Display retrieved requirements
    print("\n----------------------------------")
    print("RELEVANT REQUIREMENTS")
    print("----------------------------------")

    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if chunk_index == -1:
            continue

        print(f"\nResult {rank}")
        print(f"Similarity score: {score:.4f}")
        print("----------------------------------")
        print(chunks[chunk_index])

    print("\n----------------------------------")
    print("RETRIEVAL COMPLETED SUCCESSFULLY!")

except Exception as error:
    print("\nERROR:", error)
