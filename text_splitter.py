
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Load the research paper
pdf_path = "data/research_papers/sample.pdf"

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

print("Total pages:", len(documents))


# STEP 2: Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# STEP 3: Split the extracted document
chunks = text_splitter.split_documents(documents)


# STEP 4: Display the results
print("Total chunks generated:", len(chunks))

for i, chunk in enumerate(chunks[:3]):

    print(f"\n--- Chunk {i + 1} ---")

    print("Metadata:", chunk.metadata)

    print("Text:")
    print(chunk.page_content)