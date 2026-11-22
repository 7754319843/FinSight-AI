from pathlib import Path
import shutil

from src.document_loader import load_document
from src.chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore
from src.retriever import Retriever
from src.llm_generator import LLMGenerator
from src.metric_extractor import extract_financial_metrics


class FinSightRAG:
    """
    Main pipeline for FinSight RAG.
    """

    def __init__(self, vector_store_path: str = "vector_store"):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FaissVectorStore()
        self.llm = LLMGenerator()
        self.chunks = []
        self.vector_store_path = vector_store_path
        self.chat_history = []

    def ingest_documents(self, file_paths, save_index: bool = False):
        new_chunks = self._load_and_chunk_files(file_paths)

        texts = [chunk["text"] for chunk in new_chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        self.vector_store.build(
            embeddings=embeddings,
            chunks=new_chunks
        )

        self.chunks = new_chunks

        if save_index:
            self.vector_store.save(self.vector_store_path)

        return {
            "documents_processed": len(file_paths),
            "chunks_created": len(self.chunks),
            "sources": sorted(set(chunk["source"] for chunk in self.chunks)),
            "saved_to": self.vector_store_path if save_index else None,
            "mode": "persistent" if save_index else "temporary"
        }

    def load_existing_index(self):
        index_file = Path(self.vector_store_path) / "index.faiss"
        metadata_file = Path(self.vector_store_path) / "metadata.pkl"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                "Saved vector store not found. Add documents to the database first."
            )

        self.vector_store.load(self.vector_store_path)
        self.chunks = self.vector_store.metadata

        return self._database_stats(extra={"loaded_from": self.vector_store_path})

    def add_documents(self, file_paths):
        new_chunks = self._load_and_chunk_files(file_paths)
        new_sources = set(chunk["source"] for chunk in new_chunks)

        index_file = Path(self.vector_store_path) / "index.faiss"
        metadata_file = Path(self.vector_store_path) / "metadata.pkl"

        if index_file.exists() and metadata_file.exists():
            self.vector_store.load(self.vector_store_path)
            existing_chunks = self.vector_store.metadata
        else:
            existing_chunks = []

        kept_chunks = [
            chunk for chunk in existing_chunks
            if chunk["source"] not in new_sources
        ]

        combined_chunks = kept_chunks + new_chunks

        if not combined_chunks:
            raise ValueError("No chunks available to save.")

        texts = [chunk["text"] for chunk in combined_chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        self.vector_store.build(
            embeddings=embeddings,
            chunks=combined_chunks
        )

        self.vector_store.save(self.vector_store_path)
        self.chunks = combined_chunks

        return self._database_stats(
            extra={
                "new_or_updated_documents": len(file_paths),
                "new_chunks": len(new_chunks),
                "saved_to": self.vector_store_path
            }
        )

    def clear_database(self):
        folder = Path(self.vector_store_path)

        if folder.exists():
            shutil.rmtree(folder)

        self.vector_store = FaissVectorStore()
        self.chunks = []
        self.chat_history = []

        return {
            "status": "saved database cleared",
            "cleared_path": self.vector_store_path
        }

    def ask(self, question: str):
        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )

        lower_question = question.lower()

        if self._is_company_comparison(lower_question):
            context_chunks = self._topic_aware_company_retrieval(
                retriever=retriever,
                lower_question=lower_question
            )
        else:
            context_chunks = retriever.retrieve(
                question,
                top_k=60,
                rerank_top_n=12
            )

        history_text = ""
        for previous_question, previous_answer in self.chat_history[-10:]:
            history_text += f"Previous question: {previous_question}\n"
            history_text += f"Previous answer: {previous_answer[:800]}\n\n"

        final_question = (
            f"{history_text}"
            f"Current question: {question}"
        )

        answer = self.llm.generate(
            question=final_question,
            context_chunks=context_chunks
        )

        self.chat_history.append((question, answer))

        metrics = extract_financial_metrics(context_chunks)

        return {
            "answer": answer,
            "sources": context_chunks,
            "metrics": metrics
        }

    def summarize(self):
        if not self.chunks:
            raise ValueError("No documents ingested yet.")

        # Build a summary query that covers the most important
        # financial topics across all uploaded documents.
        summary_query = (
            "business overview revenue drivers risk factors "
            "financial performance products services strategy outlook"
        )

        retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )

        selected_chunks = retriever.retrieve(
            summary_query,
            top_k=80,
            rerank_top_n=15
        )

        answer = self.llm.generate(
            question=(
                "Provide a structured financial summary of the uploaded documents. "
                "Cover: business overview, main revenue drivers, key risk factors, "
                "and notable financial metrics. Organize by company if multiple documents are loaded."
            ),
            context_chunks=selected_chunks
        )

        metrics = extract_financial_metrics(selected_chunks)

        return {
            "summary": answer,
            "sources": selected_chunks,
            "metrics": metrics
        }

    def clear_memory(self):
        self.chat_history = []

    def _load_and_chunk_files(self, file_paths):
        all_pages = []

        for file_path in file_paths:
            pages = load_document(file_path)
            all_pages.extend(pages)

        chunks = create_chunks(all_pages)

        if not chunks:
            raise ValueError("No chunks were created. Check document text quality.")

        return chunks

    def _database_stats(self, extra=None):
        stats = {
            "documents_processed": len(set(chunk["source"] for chunk in self.chunks)),
            "chunks_created": len(self.chunks),
            "sources": sorted(set(chunk["source"] for chunk in self.chunks)),
        }

        if extra:
            stats.update(extra)

        return stats

    def _is_company_comparison(self, lower_question: str) -> bool:
        companies = ["apple", "microsoft", "nvidia"]

        comparison_words = [
            "compare",
            "across",
            "which company",
            "which one",
            "most exposed",
            "least exposed",
            "strongest",
            "weakest",
            "better",
            "worse"
        ]

        topic_words = [
            "risk",
            "revenue",
            "business model",
            "products",
            "services",
            "ai",
            "cloud",
            "data center",
            "datacenter",
            "infrastructure",
            "financial metrics"
        ]

        mentions_multiple_companies = sum(company in lower_question for company in companies) >= 2
        has_comparison_language = any(word in lower_question for word in comparison_words)
        has_financial_topic = any(word in lower_question for word in topic_words)

        return mentions_multiple_companies or (has_comparison_language and has_financial_topic)
    def _detect_topic(self, lower_question: str) -> str:
        if any(word in lower_question for word in ["risk", "risks", "risk factors", "item 1a"]):
            return "risk"

        if any(word in lower_question for word in ["revenue", "sales", "driver", "drivers", "growth"]):
            return "revenue"

        if any(word in lower_question for word in ["business model", "business models", "model"]):
            return "business_model"

        if any(word in lower_question for word in ["product", "products", "services", "offerings"]):
            return "products_services"

        if any(word in lower_question for word in ["ai", "cloud", "data center", "datacenter", "infrastructure"]):
            return "ai_cloud"

        if any(word in lower_question for word in ["financial metrics", "metrics", "margin", "income", "profit", "eps"]):
            return "financial_metrics"

        return "general"

    # Multi-document financial RAG can suffer from retrieval imbalance,
    # where one company's filing dominates retrieved chunks for comparison queries.
    # This method performs company-aware retrieval to ensure balanced evidence
    # across multiple filings before merging the final context.
    def _topic_aware_company_retrieval(self, retriever, lower_question):
        topic = self._detect_topic(lower_question)

        company_files = {
            "apple": "apple_10k_2024.pdf",
            "microsoft": "microsoft_10k_2024.pdf",
            "nvidia": "nvidia_10k_2024.pdf"
        }

        topic_queries = {
            "risk": {
                "apple": "Apple Item 1A risk factors supply chain competition cybersecurity regulatory",
                "microsoft": "Microsoft Item 1A risk factors competition regulation cybersecurity AI cloud",
                "nvidia": "NVIDIA Item 1A risk factors supply chain competition regulation AI demand"
            },
            "revenue": {
                "apple": "Apple net sales revenue drivers products services iPhone Mac iPad wearables services",
                "microsoft": "Microsoft revenue drivers cloud Azure Office LinkedIn Windows gaming server products",
                "nvidia": "NVIDIA revenue drivers data center gaming professional visualization automotive networking"
            },
            "business_model": {
                "apple": "Apple business model products services segments customers distribution",
                "microsoft": "Microsoft business model cloud productivity software gaming devices Azure Office",
                "nvidia": "NVIDIA business model GPUs data center gaming AI accelerated computing networking"
            },
            "products_services": {
                "apple": "Apple products services iPhone Mac iPad wearables accessories App Store iCloud AppleCare",
                "microsoft": "Microsoft products services Windows Microsoft 365 Azure LinkedIn Dynamics Xbox cloud",
                "nvidia": "NVIDIA products services GPUs data center gaming professional visualization automotive AI software"
            },
            "ai_cloud": {
                "apple": "Apple AI cloud services machine learning silicon services infrastructure",
                "microsoft": "Microsoft AI cloud Azure Copilot OpenAI data center infrastructure cloud services",
                "nvidia": "NVIDIA AI data center accelerated computing GPUs networking DGX cloud AI infrastructure"
            },
            "financial_metrics": {
                "apple": "Apple financial metrics net sales gross margin operating income net income fiscal 2024",
                "microsoft": "Microsoft financial metrics revenue gross margin operating income net income fiscal 2024",
                "nvidia": "NVIDIA financial metrics revenue gross margin operating income net income fiscal 2024"
            },
            "general": {
                "apple": "Apple business revenue risks products services fiscal 2024",
                "microsoft": "Microsoft business revenue risks products services fiscal 2024",
                "nvidia": "NVIDIA business revenue risks products services fiscal 2024"
            }
        }

        selected_queries = topic_queries.get(topic, topic_queries["general"])

        context_chunks = []

        mentioned_companies = [
            company for company in company_files
            if company in lower_question
        ]

        if not mentioned_companies:
            mentioned_companies = list(company_files.keys())

        for company in mentioned_companies:
            company_file = company_files[company]

            retrieved = retriever.retrieve(
                selected_queries[company],
                top_k=120,
                rerank_top_n=30
            )

            company_only = [
                chunk for chunk in retrieved
                if chunk["source"].lower() == company_file
            ]

            preferred_chunks = self._prefer_topic_sections(
                chunks=company_only,
                topic=topic
            )

            context_chunks.extend(preferred_chunks[:3])
            
        return self._remove_duplicate_chunks(context_chunks)

    def _prefer_topic_sections(self, chunks, topic):
        if topic == "risk":
            preferred_sections = ["risk_factors"]
        elif topic in ["revenue", "financial_metrics"]:
            preferred_sections = ["management_discussion", "financial_statements", "general"]
        elif topic in ["business_model", "products_services", "ai_cloud"]:
            preferred_sections = ["business_overview", "management_discussion", "general"]
        else:
            preferred_sections = [
                "business_overview",
                "management_discussion",
                "risk_factors",
                "financial_statements",
                "general"
            ]

        preferred = [
            chunk for chunk in chunks
            if chunk.get("section") in preferred_sections
        ]

        return preferred if preferred else chunks

    def _remove_duplicate_chunks(self, chunks):
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if chunk_id not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk_id)

        return unique_chunks