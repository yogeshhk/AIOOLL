"""
PrajnaAI — Retrieval-Augmented Generation (RAG) Pipeline
=========================================================
Fully local RAG using:
  - LangChain for orchestration
  - nomic-embed-text (via Ollama) for embeddings
  - ChromaDB for vector storage
  - Gemma2:2b / Qwen2:1.5b for answer generation
  - PDF and TXT document ingestion

Zero internet required after initial model download.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional

from loguru import logger

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
KB_DIR = ROOT / "knowledge_base"
PERSIST_DIR = ROOT / ".chroma_db"
PERSIST_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma2:2b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have that information in my knowledge base."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
)


# ═══════════════════════════════════════════════════════════════════════════
# RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class LocalRAGPipeline:
    """
    Fully local RAG pipeline using Ollama embeddings and LLM.
    Supports TXT and PDF document ingestion with persistent ChromaDB storage.
    """

    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        llm_model: str = LLM_MODEL,
        kb_dir: Path = KB_DIR,
        persist_dir: Path = PERSIST_DIR,
    ):
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.kb_dir = kb_dir
        self.persist_dir = persist_dir
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None

    def _check_ollama(self) -> bool:
        """Verify Ollama is running and models are available."""
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            
            missing = []
            for needed in [self.embed_model, self.llm_model]:
                if not any(needed in m for m in models):
                    missing.append(needed)
            
            if missing:
                logger.warning(f"Missing Ollama models: {missing}")
                logger.warning(f"Run: ollama pull {' '.join(missing)}")
                return False
            return True
        except Exception as e:
            logger.error(f"Ollama not reachable: {e}")
            logger.error("Start Ollama with: ollama serve")
            return False

    def load_documents(self) -> list[Document]:
        """Load all documents from knowledge base directory."""
        logger.info(f"Loading documents from {self.kb_dir}")
        
        docs = []
        for txt_file in self.kb_dir.glob("*.txt"):
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
            logger.info(f"  Loaded: {txt_file.name} ({txt_file.stat().st_size // 1024}KB)")

        # PDF support
        try:
            import pdfplumber
            for pdf_file in self.kb_dir.glob("*.pdf"):
                with pdfplumber.open(pdf_file) as pdf:
                    text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": str(pdf_file), "type": "pdf"}
                    ))
                logger.info(f"  Loaded PDF: {pdf_file.name}")
        except ImportError:
            pass

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    def build_index(self, force_rebuild: bool = False) -> None:
        """Chunk documents and build ChromaDB vector index."""
        # Check if index already exists
        if not force_rebuild and (self.persist_dir / "chroma.sqlite3").exists():
            logger.info("Loading existing vector index...")
            embeddings = OllamaEmbeddings(model=self.embed_model)
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=embeddings
            )
            logger.info(f"Index loaded: {self.vectorstore._collection.count()} chunks")
            return

        logger.info("Building new vector index...")
        docs = self.load_documents()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # Build embeddings and index
        logger.info(f"Building embeddings with {self.embed_model}...")
        t0 = time.time()
        embeddings = OllamaEmbeddings(model=self.embed_model)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.persist_dir)
        )
        elapsed = time.time() - t0
        logger.info(f"Index built in {elapsed:.2f}s | {len(chunks)} chunks indexed")

    def build_chain(self) -> None:
        """Initialize the RetrievalQA chain."""
        if self.vectorstore is None:
            raise RuntimeError("Build index first with build_index()")

        llm = Ollama(
            model=self.llm_model,
            temperature=0.1,
            num_predict=512,
        )
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
        )
        logger.info(f"QA chain ready | LLM: {self.llm_model} | Top-K: {TOP_K}")

    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline.
        Returns answer + source documents + timing.
        """
        if self.qa_chain is None:
            raise RuntimeError("Call build_chain() first")

        logger.info(f"Query: {question[:80]}...")
        t0 = time.time()
        result = self.qa_chain({"query": question})
        elapsed = time.time() - t0

        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        contexts = [doc.page_content[:200] + "..." for doc in result["source_documents"]]

        return {
            "question": question,
            "answer": answer,
            "sources": list(set(sources)),
            "contexts": contexts,
            "latency_s": elapsed,
        }

    def add_document(self, content: str, source: str = "user_input") -> int:
        """Add a new document to the existing index."""
        if self.vectorstore is None:
            raise RuntimeError("Build index first")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        doc = Document(page_content=content, metadata={"source": source})
        chunks = splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks from '{source}'")
        return len(chunks)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("🕉️  PrajnaAI — RAG Pipeline Module")
    logger.info("="*60)

    pipeline = LocalRAGPipeline()

    if not pipeline._check_ollama():
        logger.error("Please start Ollama and pull required models.")
        logger.info("  ollama serve &")
        logger.info("  ollama pull nomic-embed-text")
        logger.info("  ollama pull gemma2:2b")
        sys.exit(1)

    # Build / load index
    pipeline.build_index()
    pipeline.build_chain()

    # Demo queries
    DEMO_QUESTIONS = [
        "What models are recommended for CPU-only inference?",
        "How does GGUF quantization work?",
        "What is Ollama and how do I install it?",
        "What are the performance optimization tips for CPU inference?",
        "Which model is best for multilingual applications?",
    ]

    logger.info("\n" + "="*60)
    logger.info("DEMO QUERIES")
    logger.info("="*60)

    for q in DEMO_QUESTIONS:
        result = pipeline.query(q)
        logger.info(f"\nQ: {result['question']}")
        logger.info(f"A: {result['answer'][:300]}...")
        logger.info(f"   Sources: {result['sources']}")
        logger.info(f"   Latency: {result['latency_s']:.2f}s")
        logger.info("")

    logger.info("✅ RAG demo complete!")
    logger.info("🎨 Launch UI: streamlit run src/rag/ui/app.py")


if __name__ == "__main__":
    main()
