
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# STEP 1: Load the existing embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# STEP 2: Connect to the existing ChromaDB database
vector_database = Chroma(
    collection_name="medical_research",
    persist_directory="./vector_store",
    embedding_function=embeddings_model
)

print("Connected to ChromaDB successfully!")

print("Total stored records:",
      vector_database._collection.count())


# STEP 3: Define a retrieval function
def retrieve_documents(question, number_of_results=3):

    results = vector_database.similarity_search_with_score(
        question,
        k=number_of_results
    )

    return results


# STEP 4: Ask a research question
question = "What performance metrics were used to evaluate CNN models?"

print("\nUser question:", question)


# STEP 5: Retrieve relevant documents
retrieved_results = retrieve_documents(question)


# STEP 6: Display retrieved results
for i, (document, score) in enumerate(retrieved_results):

    print(f"\n--- Retrieved Result {i + 1} ---")

    page = document.metadata.get("page")

    if page is not None:
        print("Page:", page + 1)

    print("Source:", document.metadata.get("source"))

    print("Distance score:", round(score, 4))

    print("Content:")
    print(document.page_content)