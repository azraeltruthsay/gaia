import os
import logging
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from gaia_common.utils.knowledge_index import KnowledgeIndex

logger = logging.getLogger("GAIA.VectorStore")

class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        self.vector_store = None

    def initialize_store(self):
        """Initializes or loads the Chroma vector store."""
        try:
            os.makedirs(self.config.vectordb_path, exist_ok=True)
            self.vector_store = Chroma(
                collection_name="gaia-documents",
                embedding_function=self.embeddings,
                persist_directory=self.config.vectordb_path
            )
            logger.info("📦 Vector store initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}", exc_info=True)
            raise

    def persist(self):
        """Persists the current state of the vector store."""
        if self.vector_store:
            try:
                self.vector_store.persist()
                logger.info("💾 Vector store changes persisted to disk.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist vector store: {e}", exc_info=True)

    def delete_all_documents(self):
        """Deletes all documents from the vector store."""
        if self.vector_store:
            try:
                self.vector_store.delete(ids=None)  # Deletes all documents
                logger.info("🗑️ All documents deleted from vector store.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete documents: {e}", exc_info=True)

    def as_retriever(self):
        """Returns a retriever interface from the vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": 5})

    def add_documents(self, documents: List[Document]):
        """Adds pre-chunked documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"➕ Added {len(documents)} documents to vector store.")
        except Exception as e:
            logger.error(f"❌ Error adding documents to vector store: {e}", exc_info=True)

    def split_and_embed_documents(self, raw_documents: List[str], source: Optional[str] = None):
        """Splits raw strings into chunks and embeds them as Documents."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = splitter.create_documents(raw_documents)
            for doc in documents:
                if source:
                    doc.metadata["source"] = source
            self.add_documents(documents)
        except Exception as e:
            logger.error(f"❌ Failed to split and embed documents: {e}", exc_info=True)
