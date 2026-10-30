
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# STEP 1: Load the research paper
pdf_path = "data/research_papers/sample.pdf"

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

print("Total pages:", len(documents))


# STEP 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# STEP 3: Load the embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# STEP 4: Create the ChromaDB database
vector_database = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    collection_name="medical_research",
    persist_directory="./vector_store",
    ids=[f"research-chunk-{i}" for i in range(len(chunks))]
)

print("\nChromaDB created successfully!")


# STEP 5: Check the number of stored records
print("Total records in database:",
      vector_database._collection.count())


# STEP 6: Test a semantic search
question = "How can deep learning identify chest X-ray abnormalities?"

results = vector_database.similarity_search(
    question,
    k=3
)

print("\nTop 3 relevant research sections:")

for i, result in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print("Page:", result.metadata.get("page"))
    print("Source:", result.metadata.get("source"))
    print("Content:", result.page_content)