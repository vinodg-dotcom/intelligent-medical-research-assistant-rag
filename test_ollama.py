from langchain_ollama import ChatOllama

# Initialize the local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Send a question to the LLM
response = llm.invoke(
    "Explain Retrieval Augmented Generation "
    "with a medical research example."
)

# Display the response
print(response.content)