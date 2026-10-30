
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Medical Research Assistant API",
    description="REST API for a multi-document RAG application",
    version="1.1.0"
)


# --------------------------------------------------
# 2. Initialize Embedding Model
# --------------------------------------------------

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

vector_database = Chroma(
    collection_name="medical_research",
    persist_directory="./vector_store",
    embedding_function=embeddings_model
)


# --------------------------------------------------
# 4. Initialize Local LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# --------------------------------------------------
# 5. Request Models
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    session_id: str


# --------------------------------------------------
# 6. Conversation Memory
# --------------------------------------------------

# Stores conversation history separately for each session.
# History is cleared when the API server restarts.

chat_history: Dict[str, List[dict]] = {}


# --------------------------------------------------
# 7. Health Check API
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "running",
        "application": "Medical Research Assistant API",
        "stored_chunks": vector_database._collection.count()
    }


# --------------------------------------------------
# 8. Basic Question Answering API
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        retriever = vector_database.as_retriever(
            search_kwargs={"k": 3}
        )

        documents = retriever.invoke(question)

        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No relevant research information found."
            )

        context = ""
        sources = []

        for document in documents:

            page = document.metadata.get("page", 0) + 1
            source = document.metadata.get("source", "Unknown")

            context += f"""
            Source: {source}
            Page: {page}
            Content: {document.page_content}
            """

            sources.append({
                "document": source,
                "page": page
            })

        prompt = f"""
        You are a medical research assistant.

        Answer using ONLY the supplied research context.
        Do not invent information.
        If the answer is unavailable, clearly state that.
        Do not provide medical diagnosis or treatment advice.

        Research Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# 9. Conversational Question Answering API
# --------------------------------------------------

@app.post("/chat")
def chat_with_history(request: ChatRequest):

    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty."
        )

    try:

        # Create a new conversation if session does not exist
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Retrieve relevant research documents
        retriever = vector_database.as_retriever(
            search_kwargs={"k": 3}
        )

        documents = retriever.invoke(question)

        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No relevant research information found."
            )

        # Prepare research context and source information
        context = ""
        sources = []

        for document in documents:

            page = document.metadata.get("page", 0) + 1
            source = document.metadata.get("source", "Unknown")

            context += f"""
            Source: {source}
            Page: {page}
            Content: {document.page_content}
            """

            sources.append({
                "document": source,
                "page": page
            })

        # Retrieve previous conversation
        previous_messages = chat_history[session_id]

        history_text = ""

        for message in previous_messages:
            history_text += (
                f"\n{message['role']}: {message['content']}"
            )

        # Build conversational RAG prompt
        prompt = f"""
        You are a medical research assistant.

        Answer using ONLY the supplied research context.
        Use previous conversation to understand follow-up questions.
        Do not invent information.
        If the research context does not support the answer,
        clearly state that.
        Do not provide medical diagnosis or treatment advice.

        Previous Conversation:
        {history_text}

        Research Context:
        {context}

        Current Question:
        {question}

        Answer:
        """

        # Generate answer
        response = llm.invoke(prompt)

        answer = response.content

        # Save user question
        chat_history[session_id].append({
            "role": "user",
            "content": question
        })

        # Save assistant answer
        chat_history[session_id].append({
            "role": "assistant",
            "content": answer
        })

        return {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "conversation_messages": len(chat_history[session_id])
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# 10. Retrieve Conversation History
# --------------------------------------------------

@app.get("/chat/{session_id}")
def get_chat_history(session_id: str):

    if session_id not in chat_history:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    return {
        "session_id": session_id,
        "history": chat_history[session_id]
    }


# --------------------------------------------------
# 11. Clear Conversation History
# --------------------------------------------------

@app.delete("/chat/{session_id}")
def clear_chat_history(session_id: str):

    if session_id not in chat_history:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    del chat_history[session_id]

    return {
        "message": "Conversation history cleared successfully.",
        "session_id": session_id
    }
