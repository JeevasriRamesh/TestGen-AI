
from pathlib import Path
import os
import json
import faiss

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# Find project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# File locations
INDEX_FILE = PROJECT_FOLDER / "output" / "faiss_index.bin"
CHUNKS_FILE = PROJECT_FOLDER / "output" / "chunks.json"
OUTPUT_FILE = PROJECT_FOLDER / "output" / "generated_test_cases.txt"

print("TESTGEN AI - TEST CASE GENERATION")
print("----------------------------------")


try:
    # Step 1: Load API key
    print("\nStep 1: Loading Groq API key...")

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Groq API key not found in .env file.")

    client = Groq(api_key=api_key)

    print("Groq client created successfully!")

    # Step 2: Load FAISS index and chunks
    print("\nStep 2: Loading requirement database...")

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Required FAISS files are missing. Run create_embeddings.py first."
        )

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Requirement database loaded!")

    # Step 3: Load embedding model
    print("\nStep 3: Loading embedding model...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!")

    # Step 4: Get user question
    question = input(
        "\nEnter your testing requirement: "
    ).strip()

    if not question:
        raise ValueError("Please enter a requirement.")

    # Step 5: Retrieve relevant requirements
    print("\nStep 4: Retrieving relevant requirements...")

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    number_of_results = min(3, len(chunks))

    scores, indices = index.search(
        question_embedding,
        number_of_results
    )

    retrieved_chunks = []

    for chunk_index in indices[0]:
        if chunk_index != -1:
            retrieved_chunks.append(chunks[chunk_index])

    context = "\n\n".join(retrieved_chunks)

    print("Relevant requirements retrieved!")

    # Step 6: Prepare prompt for Groq
    print("\nStep 5: Preparing AI prompt...")

    prompt = f"""
You are an experienced Software Quality Assurance Engineer.

Your task is to generate software test cases using the
user's question and the retrieved software requirements.

USER QUESTION:
{question}

RETRIEVED REQUIREMENTS:
{context}

INSTRUCTIONS:

1. Generate relevant test cases based on the retrieved requirements.
2. Include positive and negative test scenarios.
3. Include boundary or validation scenarios wherever supported.
4. Do not invent features that are not mentioned in the requirements.
5. If any detail is missing, clearly mention it as an assumption.
6. Make every test case clear and suitable for a beginner tester.

For each test case, provide:

Test Case ID:
Test Scenario:
Test Type:
Preconditions:
Test Steps:
Test Data:
Expected Result:
Priority:

Generate at least 5 useful test cases.

Present the result in a clean, readable format.
"""

    # Step 7: Send request to Groq
    print("\nStep 6: Generating test cases using Groq AI...")
    print("Please wait...")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=2000
    )

    generated_text = response.choices[0].message.content

    if not generated_text:
        raise ValueError("Groq returned an empty response.")

    # Step 8: Display generated test cases
    print("\n----------------------------------")
    print("GENERATED TEST CASES")
    print("----------------------------------")

    print(generated_text)

    # Step 9: Save generated test cases
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    OUTPUT_FILE.write_text(
        generated_text,
        encoding="utf-8"
    )

    print("\n----------------------------------")
    print("TEST CASE GENERATION COMPLETED!")
    print("----------------------------------")

    print("Saved file:", OUTPUT_FILE)

except Exception as error:
    print("\nERROR: Test case generation failed!")
    print(error)
