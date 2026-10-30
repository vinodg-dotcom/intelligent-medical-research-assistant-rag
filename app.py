
import streamlit as st
import tempfile
import os
import hashlib
import uuid
import requests

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# 2. Initialize Embedding Model and ChromaDB
# --------------------------------------------------

@st.cache_resource
def load_components():

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_database = Chroma(
        collection_name="medical_research",
        persist_directory="./vector_store",
        embedding_function=embeddings_model
    )

    return vector_database


try:
    vector_database = load_components()

except Exception as error:
    st.error(f"Initialization failed: {error}")
    st.stop()


# --------------------------------------------------
# 3. Initialize Session State
# --------------------------------------------------

# Generate a unique ID for this conversation.

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# Store messages displayed in the Streamlit interface.

if "messages" not in st.session_state:
    st.session_state.messages = []


# Avoid processing the same PDF repeatedly in this session.

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# --------------------------------------------------
# 4. Application Header
# --------------------------------------------------

st.title("🔬 Medical Research Assistant")

st.write(
    "Upload medical research papers and ask questions "
    "using your local AI research assistant."
)

st.info(
    "Research support only. "
    "Not intended for medical diagnosis or treatment recommendations."
)


# --------------------------------------------------
# 5. PDF Upload and Processing
# --------------------------------------------------

st.header("📄 Upload Research Papers")

uploaded_files = st.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("Process Uploaded PDFs"):

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in st.session_state.processed_files:

                st.warning(
                    f"{uploaded_file.name} was already processed."
                )

                continue

            temp_path = None

            try:

                # Save uploaded PDF temporarily.

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False
                ) as temp_file:

                    temp_file.write(file_bytes)
                    temp_path = temp_file.name


                # Extract text from PDF.

                loader = PyMuPDFLoader(temp_path)

                documents = loader.load()


                # Preserve original PDF filename.

                for document in documents:

                    document.metadata["source"] = uploaded_file.name


                # Split extracted text into chunks.

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = text_splitter.split_documents(documents)


                if not chunks:

                    st.warning(
                        f"No extractable text found in {uploaded_file.name}."
                    )

                    continue


                # Generate unique IDs for chunks.

                chunk_ids = [
                    str(uuid.uuid4())
                    for _ in chunks
                ]


                # Store chunks in ChromaDB.

                vector_database.add_documents(
                    documents=chunks,
                    ids=chunk_ids
                )


                st.session_state.processed_files.add(file_hash)

                st.success(
                    f"{uploaded_file.name}: "
                    f"{len(documents)} pages, "
                    f"{len(chunks)} chunks stored."
                )


            except Exception as error:

                st.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )


            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


# --------------------------------------------------
# 6. Vector Database Status
# --------------------------------------------------

st.divider()

st.subheader("📊 Vector Database Status")

total_chunks = vector_database._collection.count()

st.metric(
    label="Total stored document chunks",
    value=total_chunks
)


# --------------------------------------------------
# 7. Chat History Display
# --------------------------------------------------

st.divider()

st.header("💬 Ask Your Research Assistant")

# Display previous questions and answers.

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.markdown("**Retrieved Sources**")

                for source in sources:

                    document_name = source.get(
                        "document",
                        "Unknown document"
                    )

                    page_number = source.get(
                        "page",
                        "Unknown"
                    )

                    st.write(
                        f"📄 {document_name} – Page {page_number}"
                    )


# --------------------------------------------------
# 8. Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded research papers..."
)


if question:

    # Display user's question immediately.

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    with st.chat_message("user"):

        st.markdown(question)


    # Generate answer using FastAPI.

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching research papers and generating answer..."
        ):

            try:

                if total_chunks == 0:

                    st.warning(
                        "Please upload and process a research paper first."
                    )

                else:

                    api_url = "http://127.0.0.1:8000/chat"


                    # Send question and session ID to FastAPI.

                    response = requests.post(
                        api_url,
                        json={
                            "question": question,
                            "session_id": st.session_state.session_id
                        },
                        timeout=180
                    )


                    response.raise_for_status()

                    result = response.json()


                    answer = result["answer"]

                    sources = result.get("sources", [])


                    # Display generated answer.

                    st.markdown(answer)


                    # Display retrieved sources.

                    if sources:

                        st.markdown("**Retrieved Sources**")

                        for source in sources:

                            document_name = source.get(
                                "document",
                                "Unknown document"
                            )

                            page_number = source.get(
                                "page",
                                "Unknown"
                            )

                            st.write(
                                f"📄 {document_name} – Page {page_number}"
                            )


                    # Save assistant response in Streamlit history.

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Please ensure the API server is running."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )


            except requests.exceptions.HTTPError as error:

                st.error(
                    f"FastAPI returned an HTTP error: {error}"
                )


            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )


# --------------------------------------------------
# 9. Sidebar - Project Information
# --------------------------------------------------

with st.sidebar:

    st.header("Project Information")

    st.write("**Embedding Model:**")
    st.write("all-MiniLM-L6-v2")

    st.write("**Vector Database:**")
    st.write("ChromaDB")

    st.write("**Language Model:**")
    st.write("Llama 3.2 3B")

    st.write("**Runtime:**")
    st.write("Ollama")

    st.write("**Backend API:**")
    st.write("FastAPI")

    st.write("**Conversation Memory:**")
    st.write("Session-based")

    st.divider()

    st.caption(
        "Medical Research Assistant – Multi-PDF RAG Project"
    )

    st.caption(
        f"Session ID: {st.session_state.session_id[:8]}..."
    )

    # Reset conversation.

    if st.button("Clear Chat History"):

        try:

            requests.delete(
                f"http://127.0.0.1:8000/chat/{st.session_state.session_id}",
                timeout=30
            )

        except requests.exceptions.RequestException:

            pass

        st.session_state.messages = []

        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()
