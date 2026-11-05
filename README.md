TestGen AI – AI-Powered Software Test Case Generator Using RAG

TestGen AI is an AI-powered software testing application that generates relevant software test cases from Software Requirements Specification (SRS) PDF documents using Retrieval-Augmented Generation (RAG).Users can upload an SRS document, enter a testing requirement, and receive AI-generated test cases covering positive, negative, boundary and validation scenarios. The generated test cases can also be downloaded as an Excel spreadsheet.


Features

PDF Upload: Upload SRS documents in PDF format.

Text Extraction: Extract requirements from uploaded PDF documents using PyPDF.

Text Chunking: Split extracted requirements into smaller chunks using LangChain.

Text Embeddings: Convert requirement chunks into numerical embeddings using Sentence Transformers.

Semantic Search: Retrieve relevant requirement chunks using FAISS similarity search.

AI Test Case Generation: Generate test cases using Groq AI.

Multiple Test Scenarios: Generate positive, negative, boundary and validation test cases.

Excel Export: Download generated test cases in a formatted Excel spreadsheet.

Interactive Interface: Simple and beginner-friendly interface built with Streamlit.



Live Demo: https://testgen-aigit-tvwsscfa7b4debjmi7pur4.streamlit.app/
