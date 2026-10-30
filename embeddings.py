
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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


# STEP 3: Initialize the embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# STEP 4: Convert text chunks into vectors
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings_model.embed_documents(texts)


# STEP 5: Validate the embeddings
print("\nTotal vectors generated:", len(vectors))

print("Dimensions of first vector:", len(vectors[0]))

print("\nFirst 10 numerical values:")
print(vectors[0][:10])


# STEP 6: Test question embedding
question = "How can deep learning identify chest X-ray abnormalities?"

question_vector = embeddings_model.embed_query(question)

print("\nQuestion vector dimensions:", len(question_vector))