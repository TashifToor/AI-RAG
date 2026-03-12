import os
import uuid
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()

# ── 1. Text Chunker ──────────────────────────────────────────────────────────
class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)

# ── 2. Embedding Manager ─────────────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

# ── 3. Vector Store ──────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, collection_name="pdf_document", persist_directory="./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.__initialize_store() 

    def __initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Pdf Document embedding for RAG"}
        )

    def add_document(self, documents: List[Any], embeddings: np.ndarray):
        ids, metadatas, documents_text, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

# ── 4. RAG Retrieval ─────────────────────────────────────────────────────────
class RAGRetrieval:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        query_embedding = self.embedding_manager.generate_embedding([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        retrieved_docs = []
        if results["documents"] and results['documents'][0]:
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0],
                results['metadatas'][0], results['distances'][0]
            )):
                similarity_score = 1 / (1 + distance)
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1,
                    })
        return retrieved_docs

# ── 5. RAG Function ──────────────────────────────────────────────────────────
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        return "No Relevant Context Found to answer the question"
    prompt = f"""You are a professional CV assistant of me so act like you are the CV. Answer questions about the candidate's background, skills, and experience in a clear and professional tone.
Never say phrases like "based on the context", "according to the document", or "the context mentions".
Answer directly and professionally as if you are presenting the candidate's profile.
If the answer is not available, simply say "I don't have that information."


Additional Information about the candidate:
    - Currently in 2nd semester of IT
    - Building RAG systems using LangChain, ChromaDB and Groq
    - Studying and practicing Machine Learning including supervised and unsupervised learning
    - And in experience say i have backend experience at CodeCelix and Ai/ml Experience at World wise Solution like rag llm lanchain you format it by yourself
    - Hands on experience with LLMs and AI Application Development and Professional in Backend Development
    

    CV Information:
    {context}

    Question:
    {query}

    Answer:
    """
    response = llm.invoke([prompt])
    return response.content

# ── 6. Load Everything Once ──────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with st.spinner("Loading PDFs and building vector store..."):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
        VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_store")

        loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()

        if not documents:
            st.error("No PDFs found!")
            st.stop()

        chunker = TextChunker()
        chunks = chunker.split_documents(documents)

        embedding_manager = EmbeddingManager()
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embedding(texts)

        vector_store = VectorStore(persist_directory=VECTOR_DIR)
        vector_store.add_document(chunks, embeddings)

        retriever = RAGRetrieval(embedding_manager, vector_store)

        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )
    return retriever, llm
# ── 7. Streamlit UI ──────────────────────────────────────────────────────────
st.title("📄 RAG Chatbot")
st.caption("Ask anything from your PDF documents")

retriever, llm = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_simple(prompt, retriever, llm)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})