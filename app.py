
import os
import re
import html
from io import BytesIO

import streamlit as st
import pandas as pd
import faiss

from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TestGen AI",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 TestGen AI")
st.subheader("AI-Powered Software Test Case Generator")

st.write(
    "Upload your Software Requirements Specification (SRS) PDF "
    "and generate relevant software test cases using RAG and AI."
)

st.divider()


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(PROJECT_FOLDER, ".env"))

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# --------------------------------------------------
# CLEAN GENERATED TABLE
# --------------------------------------------------

def clean_cell(value):

    value = html.unescape(value)

    value = re.sub(
        r"<br\s*/?>",
        "\n",
        value,
        flags=re.IGNORECASE
    )

    value = re.sub(r"<[^>]+>", "", value)

    value = value.replace("**", "")
    value = value.replace("*", "")

    return value.strip()


# --------------------------------------------------
# EXTRACT TEST CASES FROM AI RESPONSE
# --------------------------------------------------



def split_markdown_row(line):
    """
    Split one Markdown table row into cells.
    Handles optional outer pipes and escaped pipe characters.
    """
    line = line.strip()

    if line.startswith("|"):
        line = line[1:]

    if line.endswith("|"):
        line = line[:-1]

    # Protect escaped pipe characters before splitting.
    placeholder = "__ESCAPED_PIPE__"
    line = line.replace(r"\|", placeholder)

    cells = [cell.strip() for cell in line.split("|")]

    return [cell.replace(placeholder, "|") for cell in cells]


def extract_test_cases(generated_text):

    if not generated_text:
        return None

    # Remove optional Markdown code fences around the table.
    text = re.sub(r"```(?:markdown|md|text)?", "", generated_text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    lines = text.replace("\r", "").splitlines()

    header = None
    test_cases = []
    expected_columns = 0

    for line in lines:

        line = line.strip()

        # Find the Markdown table header even if it has leading spaces.
        if "|" not in line:
            continue

        cells = split_markdown_row(line)

        cleaned_cells = [clean_cell(cell) for cell in cells]
        normalized_cells = [
            re.sub(
                r"\s+",
                " ",
                re.sub(r"[^a-z0-9]+", " ", cell.lower())
            ).strip()
            for cell in cleaned_cells
        ]

        # Identify the header row.
        if (
            any("test case id" in cell for cell in normalized_cells)
            and any("test scenario" in cell for cell in normalized_cells)
        ):
            header = cleaned_cells
            expected_columns = len(header)
            continue

        if header is None:
            continue

        # Ignore Markdown separator rows such as |---|:---:|.
        if cells and all(
            re.fullmatch(r":?[- ]{3,}:?", cell.replace(" ", ""))
            for cell in cells
        ):
            continue

        # Skip blank/non-data lines.
        if not any(cells):
            continue

        # If a cell value itself contains an unescaped pipe, merge
        # extra pieces into the final column instead of discarding the row.
        if len(cells) > expected_columns:
            cells = (
                cells[:expected_columns - 1]
                + [" | ".join(cells[expected_columns - 1:])]
            )

        # If the model omitted trailing empty cells, pad the row.
        elif len(cells) < expected_columns:
            cells.extend([""] * (expected_columns - len(cells)))

        cleaned_row = [clean_cell(cell) for cell in cells]

        # Accept IDs such as TC-01, TC_01, TC_LOGIN_01 and TC-PAYMENT-01.
        test_id = cleaned_row[0].strip().strip("`*_ ")

        if re.fullmatch(
            r"TC(?:[-_][A-Z0-9]+)+",
            test_id,
            flags=re.IGNORECASE
        ):
            cleaned_row[0] = test_id
            test_cases.append(cleaned_row)

    if header is None or not test_cases:
        return None

    # Make column names safe and non-empty.
    cleaned_header = []
    for index, column in enumerate(header, start=1):
        name = clean_cell(column).strip()
        cleaned_header.append(name if name else f"Column {index}")

    # Ensure unique column names for pandas.
    seen = {}
    unique_header = []
    for name in cleaned_header:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_header.append(name if count == 0 else f"{name}_{count + 1}")

    dataframe = pd.DataFrame(test_cases, columns=unique_header)

    # Remove completely empty rows and duplicate test case IDs.
    dataframe = dataframe.dropna(how="all")
    dataframe = dataframe.drop_duplicates(subset=[unique_header[0]], keep="first")

    return dataframe if not dataframe.empty else None


# --------------------------------------------------
# CREATE EXCEL FILE
# --------------------------------------------------

def create_excel(dataframe):

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Test Cases"
        )

        worksheet = writer.sheets["Test Cases"]

        from openpyxl.styles import (
            Font,
            PatternFill,
            Alignment
        )

        from openpyxl.utils import get_column_letter

        # Format header
        for cell in worksheet[1]:

            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="17365D"
            )

            cell.font = Font(
                bold=True,
                color="FFFFFF"
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        # Format data
        for row in worksheet.iter_rows(min_row=2):

            for cell in row:

                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True
                )

        # Set column widths
        for column_number, column_name in enumerate(
            dataframe.columns,
            start=1
        ):

            letter = get_column_letter(column_number)

            worksheet.column_dimensions[letter].width = 35

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

    excel_buffer.seek(0)

    return excel_buffer.getvalue()


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.header("1. Upload Requirements PDF")

uploaded_file = st.file_uploader(
    "Choose your SRS PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

st.header("2. Enter Your Testing Requirement")

question = st.text_area(
    "What would you like to test?",
    placeholder=(
        "Example: What should happen when a user "
        "enters an incorrect password during login?"
    ),
    height=100
)


# --------------------------------------------------
# GENERATE TEST CASES
# --------------------------------------------------

if st.button(
    "Generate Test Cases",
    type="primary",
    use_container_width=True
):

    if not uploaded_file:

        st.error("Please upload a requirements PDF first.")

    elif not question.strip():

        st.error("Please enter a testing requirement.")

    elif not GROQ_API_KEY:

        st.error(
            "Groq API key not found. Check your .env file."
        )

    else:

        try:

            # Step 1: Extract PDF text

            with st.spinner("Reading your requirements PDF..."):

                pdf_reader = PdfReader(
                    BytesIO(uploaded_file.getvalue())
                )

                extracted_text = ""

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    page_text = page.extract_text() or ""

                    extracted_text += (
                        f"\n--- PAGE {page_number} ---\n"
                        + page_text
                    )

                if not extracted_text.strip():
                    raise ValueError(
                        "No readable text found in this PDF."
                    )

            st.success(
                f"PDF processed successfully! "
                f"Total pages: {len(pdf_reader.pages)}"
            )

            # Step 2: Create chunks

            with st.spinner("Splitting requirements into chunks..."):

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )

                chunks = splitter.split_text(
                    extracted_text
                )

            st.success(
                f"Created {len(chunks)} requirement chunks."
            )

            # Step 3: Load embedding model

            with st.spinner("Loading the embedding model..."):

                model = load_embedding_model()

            # Step 4: Create embeddings and FAISS index

            with st.spinner("Creating requirement embeddings..."):

                embeddings = model.encode(
                    chunks,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                embeddings = embeddings.astype("float32")

                index = faiss.IndexFlatIP(
                    embeddings.shape[1]
                )

                index.add(embeddings)

            # Step 5: Retrieve relevant requirements

            with st.spinner("Searching relevant requirements..."):

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

                        retrieved_chunks.append(
                            chunks[chunk_index]
                        )

                context = "\n\n".join(
                    retrieved_chunks
                )

            # Step 6: Connect to Groq

            client = Groq(
                api_key=GROQ_API_KEY
            )

            # Step 7: Prepare AI prompt

            prompt = f"""
You are an experienced Software Quality Assurance Engineer.

Generate software test cases using only the
user question and retrieved requirements below.

USER QUESTION:
{question}

RETRIEVED REQUIREMENTS:
{context}

INSTRUCTIONS:

1. Generate at least 5 useful test cases.
2. Include positive and negative scenarios.
3. Include boundary and validation tests where supported.
4. Do not present unsupported behaviour as a confirmed requirement.
5. Clearly label additional suggestions and assumptions.
6. Keep every test case relevant to the user's question.
7. Do not invent features.

For each test case provide:

Test Case ID
Test Scenario
Test Type
Preconditions
Test Steps
Test Data
Expected Result
Priority

Return the test cases as a Markdown table.
"""

            # Step 8: Generate test cases

            with st.spinner(
                "Groq AI is generating your test cases..."
            ):

                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=2500
                )

                generated_text = (
                    response.choices[0].message.content
                )

                if not generated_text:
                    raise ValueError(
                        "Groq returned an empty response."
                    )

            # Step 9: Extract test case table

            dataframe = extract_test_cases(
                generated_text
            )

            # Save results in Streamlit session

            st.session_state["generated_text"] = generated_text
            st.session_state["test_dataframe"] = dataframe
            st.session_state["retrieved_context"] = context

            if dataframe is not None:

                st.session_state["excel_file"] = create_excel(
                    dataframe
                )

            else:

                st.session_state["excel_file"] = None

            st.success(
                "Test cases generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )


# --------------------------------------------------
# DISPLAY GENERATED RESULTS
# --------------------------------------------------

if "generated_text" in st.session_state:

    st.divider()

    st.header("3. Generated Test Cases")

    st.markdown(
        st.session_state["generated_text"]
    )
    with st.expander("Show Raw AI Response"):
        st.text(st.session_state["generated_text"])
    

    dataframe = st.session_state.get(
        "test_dataframe"
    )

    if dataframe is not None:

        st.subheader("Test Case Summary")

        st.metric(
            "Total Test Cases",
            len(dataframe)
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        excel_file = st.session_state.get(
            "excel_file"
        )

        if excel_file:

            st.download_button(
                label="Download Test Cases as Excel",
                data=excel_file,
                file_name="TestGen_AI_Test_Cases.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True
            )

    else:

        st.warning(
            "AI generated a response, but its test case "
            "table could not be extracted for Excel export."
        )

    with st.expander("View Retrieved Requirements"):

        st.write(
            st.session_state["retrieved_context"]
        )


st.divider()

st.caption(
    "TestGen AI | RAG-Based AI Software Test Case Generator"
)
