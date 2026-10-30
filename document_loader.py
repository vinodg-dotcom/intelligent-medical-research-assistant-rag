from langchain_community.document_loaders import PyMuPDFLoader

# PDF file location
pdf_path = "data/research_papers/sample.pdf"

# Initialize PDF loader
loader = PyMuPDFLoader(pdf_path)

# Load PDF document
documents = loader.load()

# Display document information
print("Total pages:", len(documents))

for i, doc in enumerate(documents[:3]):

    print("\nPage:", i + 1)

    print("Metadata:", doc.metadata)

    print("Extracted text:")
    print(doc.page_content[:500])