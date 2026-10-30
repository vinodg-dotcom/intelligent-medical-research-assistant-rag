
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# STEP 1: Load the embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# STEP 2: Connect to the existing ChromaDB
vector_database = Chroma(
    collection_name="medical_research",
    persist_directory="./vector_store",
    embedding_function=embeddings_model
)


# STEP 3: Configure the Retriever
retriever = vector_database.as_retriever(
    search_kwargs={"k": 3}
)


# STEP 4: Connect to the local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# STEP 5: Define the RAG function
def ask_research_assistant(question):

    # Retrieve relevant document chunks
    retrieved_documents = retriever.invoke(question)

    # Prepare the context with page references
    context = ""

    for document in retrieved_documents:

        page = document.metadata.get("page", 0) + 1

        context += (
            f"\nSource: {document.metadata.get('source')}"
            f"\nPage: {page}"
            f"\nContent: {document.page_content}\n"
        )

    # Create the prompt
    prompt = f"""
    You are a medical research assistant.

    Answer the user's question using ONLY the
    research paper information provided below.

    Instructions:
    1. Use simple English.
    2. Do not invent information.
    3. If the answer is not available in the
       provided context, say so clearly.
    4. Mention the relevant page numbers.
    5. Do not provide medical diagnosis or
       treatment recommendations.

    Research Paper Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # Generate the answer using Llama 3.2
    response = llm.invoke(prompt)

    return response.content, retrieved_documents


# STEP 6: Ask a question
question = "What performance metrics were used to evaluate CNN models?"

answer, sources = ask_research_assistant(question)


# STEP 7: Display the answer
print("\n========== RAG ASSISTANT ==========")

print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(answer)

print("\n========== SOURCE REFERENCES ==========")

for source in sources:

    page = source.metadata.get("page", 0) + 1

    print(
        f"Source: {source.metadata.get('source')} "
        f"| Page: {page}"
    )