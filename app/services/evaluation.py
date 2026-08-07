import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

from app.core.config import TOP_K
from app.services.reteriver import RetrieverService


class EvaluationService:
    """
    Lightweight evaluation service for RAG retrieval quality.

    Metrics included:
    - Precision@k
    - Recall@k
    - MRR@k
    - Answer match
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        answer_fn: Optional[Callable[[Dict[str, Any], List[Document]], str]] = None,
    ):
        self.retriever = retriever or RetrieverService()
        self.answer_fn = answer_fn or self._default_answer_fn

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        k: int = TOP_K,
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset of questions against retrieval results.

        Expected item shape:
        {
            "question": str,
            "expected_sources": [str, ...],
            "expected_answer": str | None,
        }
        """

        if not dataset:
            return {
                "num_items": 0,
                "avg_precision_at_k": 0.0,
                "avg_recall_at_k": 0.0,
                "avg_mrr_at_k": 0.0,
                "avg_answer_match": 0.0,
                "breakdown": [],
            }

        breakdown: List[Dict[str, Any]] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        mrr_scores: List[float] = []
        answer_scores: List[float] = []

        for item in dataset:
            question = item.get("question", "")
            expected_sources = item.get("expected_sources", []) or []
            expected_answer = item.get("expected_answer")

            retrieved_docs = self.retriever.retrieve(question, k=k)
            source_names = [self._doc_source(doc) for doc in retrieved_docs]

            precision = self._precision_at_k(expected_sources, source_names, k)
            recall = self._recall_at_k(expected_sources, source_names, k)
            mrr = self._mrr_at_k(expected_sources, source_names)

            predicted_answer = self.answer_fn(item, retrieved_docs)
            answer_match = self._answer_match(predicted_answer, expected_answer)

            breakdown.append(
                {
                    "question": question,
                    "precision_at_k": round(precision, 4),
                    "recall_at_k": round(recall, 4),
                    "mrr_at_k": round(mrr, 4),
                    "answer_match": round(answer_match, 4),
                    "retrieved_sources": source_names,
                }
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            answer_scores.append(answer_match)

        return {
            "num_items": len(dataset),
            "avg_precision_at_k": round(sum(precision_scores) / len(precision_scores), 4),
            "avg_recall_at_k": round(sum(recall_scores) / len(recall_scores), 4),
            "avg_mrr_at_k": round(sum(mrr_scores) / len(mrr_scores), 4),
            "avg_answer_match": round(sum(answer_scores) / len(answer_scores), 4),
            "breakdown": breakdown,
        }

    def _default_answer_fn(self, item: Dict[str, Any], docs: List[Document]) -> str:
        if not docs:
            return ""

        text = "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))
        return text[:1000]

    def _doc_source(self, doc: Document) -> str:
        return str(doc.metadata.get("source", ""))

    def _precision_at_k(self, expected_sources: List[str], retrieved_sources: List[str], k: int) -> float:
        if not expected_sources:
            return 0.0

        relevant_hits = 0
        for source in retrieved_sources[:k]:
            if source in expected_sources:
                relevant_hits += 1

        return relevant_hits / min(k, len(expected_sources))

    def _recall_at_k(self, expected_sources: List[str], retrieved_sources: List[str], k: int) -> float:
        if not expected_sources:
            return 0.0

        relevant_hits = 0
        for source in retrieved_sources[:k]:
            if source in expected_sources:
                relevant_hits += 1

        return relevant_hits / len(expected_sources)

    def _mrr_at_k(self, expected_sources: List[str], retrieved_sources: List[str]) -> float:
        if not expected_sources:
            return 0.0

        for index, source in enumerate(retrieved_sources, start=1):
            if source in expected_sources:
                return 1.0 / index

        return 0.0

    def _answer_match(self, predicted_answer: str, expected_answer: Optional[str]) -> float:
        if not expected_answer:
            return 0.0

        normalized_pred = self._normalize_text(predicted_answer)
        normalized_expected = self._normalize_text(expected_answer)

        if not normalized_pred or not normalized_expected:
            return 0.0

        if normalized_expected in normalized_pred:
            return 1.0

        return 0.0

    def _normalize_text(self, value: str) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip().lower()
