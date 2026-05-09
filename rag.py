"""
rag.py — Retrieval-Augmented Generation Module
=================================================
Implements TF-IDF based retrieval over advisory documents.

For a research prototype, TF-IDF provides:
  - Interpretable retrieval scores
  - No external API dependencies
  - Reproducible results
  - Sufficient quality for domain-specific short documents

Components:
  - Document indexing from dataset
  - Query → Top-K retrieval
  - Context formatting for LLM prompt
"""

import math
import re
from typing import List, Dict, Tuple
from collections import Counter
from dataset import get_all_documents, get_kg_context_documents


class TFIDFRetriever:
    """TF-IDF based document retriever for agricultural advisories."""

    def __init__(self):
        self.documents: List[Dict] = []
        self.doc_vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        # Remove very short tokens and stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'and', 'but', 'or', 'nor', 'not', 'so',
            'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
            'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'if',
            'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'how', 'when', 'where', 'why',
        }
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency (normalized)."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in counts.items()}

    def index_documents(self, documents: List[Dict]):
        """Index a list of documents for retrieval."""
        self.documents = documents

        # Tokenize all documents
        all_tokens = []
        for doc in documents:
            tokens = self._tokenize(doc["text"])
            all_tokens.append(tokens)
            self.vocab.update(tokens)

        # Compute IDF
        n_docs = len(documents)
        doc_freq = Counter()
        for tokens in all_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self.idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1
            for term, df in doc_freq.items()
        }

        # Compute TF-IDF vectors
        self.doc_vectors = []
        for tokens in all_tokens:
            tf = self._compute_tf(tokens)
            tfidf = {term: tf_val * self.idf.get(term, 0)
                      for term, tf_val in tf.items()}
            self.doc_vectors.append(tfidf)

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-K most relevant documents for a query.

        Returns list of dicts with:
          - id, text, score, rank
        """
        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)
        query_vec = {term: tf_val * self.idf.get(term, 0)
                     for term, tf_val in query_tf.items()}

        # Score all documents
        scored = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            scored.append((i, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        results = []
        for rank, (idx, score) in enumerate(scored[:top_k], 1):
            results.append({
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
                "score": round(score, 4),
                "rank": rank,
                "metadata": self.documents[idx].get("metadata", {}),
            })

        return results


class RAGSystem:
    """
    Complete RAG pipeline combining TF-IDF retrieval with context formatting.
    Optionally integrates with Knowledge Graph for enhanced retrieval.
    """

    def __init__(self, use_kg_context: bool = False):
        self.retriever = TFIDFRetriever()
        self.use_kg_context = use_kg_context
        self._build_index()

    def _build_index(self):
        """Build the retrieval index from dataset."""
        # Core advisory documents
        docs = get_all_documents()

        # Optionally add KG context documents (crop/soil/weather descriptions)
        if self.use_kg_context:
            kg_docs = get_kg_context_documents()
            docs.extend(kg_docs)

        self.retriever.index_documents(docs)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def format_retrieved_context(self, results: List[Dict]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not results:
            return "No relevant documents found."

        parts = ["=== RETRIEVED DOCUMENTS (RAG) ===\n"]
        for r in results:
            parts.append(
                f"[Rank {r['rank']}] (Relevance: {r['score']:.4f})\n"
                f"{r['text']}\n"
            )
        return "\n".join(parts)

    def get_retrieval_stats(self, results: List[Dict]) -> Dict:
        """Compute retrieval statistics for evaluation."""
        if not results:
            return {"avg_score": 0, "max_score": 0, "min_score": 0, "num_results": 0}

        scores = [r["score"] for r in results]
        return {
            "avg_score": round(sum(scores) / len(scores), 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "num_results": len(results),
        }


# ──────────────────── Main (demo) ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RAG SYSTEM — Retrieval Demo")
    print("=" * 60)

    # Test without KG context
    rag_basic = RAGSystem(use_kg_context=False)
    print(f"\nIndexed {len(rag_basic.retriever.documents)} advisory documents")

    test_queries = [
        "What irrigation advice for wheat during drought?",
        "How to manage pests in rice during monsoon?",
        "Fertilizer for maize in red soil",
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        results = rag_basic.retrieve_context(query, top_k=2)
        print(rag_basic.format_retrieved_context(results))
        stats = rag_basic.get_retrieval_stats(results)
        print(f"Stats: {stats}")

    # Test with KG context
    print(f"\n{'=' * 60}")
    print("RAG + KG Context — Enhanced Retrieval")
    print("=" * 60)

    rag_enhanced = RAGSystem(use_kg_context=True)
    print(f"\nIndexed {len(rag_enhanced.retriever.documents)} documents (advisory + KG context)")

    query = "What irrigation advice for wheat during drought?"
    results = rag_enhanced.retrieve_context(query, top_k=3)
    print(f"\nQuery: {query}")
    print(rag_enhanced.format_retrieved_context(results))
