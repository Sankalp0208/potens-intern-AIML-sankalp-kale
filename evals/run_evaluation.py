import json
import time
from pathlib import Path

import requests

from evals.dataset import EVAL_DATASET


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

ASK_URL = f"{BASE_URL}/ask"

RESULTS_DIR = Path("evals/results")
RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"


# ============================================================
# CALL ACTUAL RAG /ASK ENDPOINT
# ============================================================

def call_rag(question: str) -> dict:

    payload = {
        "question": question
    }

    try:

        response = requests.post(
            ASK_URL,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:

        return {
            "answer": "",
            "confidence": 0.0,
            "citations": [],
            "error": str(e),
        }


# ============================================================
# EXTRACT SOURCES FROM CITATIONS
# ============================================================

def extract_sources(citations):

    sources = []

    for citation in citations:

        source = citation.get("source")

        if source:
            sources.append(source)

    return sources


# ============================================================
# EXTRACT CONTEXT FROM CITATION SNIPPETS
# ============================================================

def extract_contexts(citations):

    contexts = []

    for citation in citations:

        snippet = citation.get("snippet")

        if snippet:
            contexts.append(snippet)

    return contexts


# ============================================================
# NORMALIZE SOURCE NAME
# ============================================================

def normalize_source(source):

    if not source:
        return ""

    return Path(str(source)).name.lower().strip()


# ============================================================
# CITATION RECALL
# ============================================================

def citation_recall(expected_sources, actual_sources):

    if not expected_sources:
        return None

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    actual = {
        normalize_source(source)
        for source in actual_sources
    }

    matched = expected.intersection(actual)

    return len(matched) / len(expected)


# ============================================================
# CITATION PRECISION
# ============================================================

def citation_precision(expected_sources, actual_sources):

    if not expected_sources:
        return None

    if not actual_sources:
        return 0.0

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    actual = [
        normalize_source(source)
        for source in actual_sources
    ]

    relevant = sum(
        1
        for source in actual
        if source in expected
    )

    return relevant / len(actual)


# ============================================================
# EXACT SOURCE MATCH
# ============================================================

def source_match(expected_sources, actual_sources):

    if not expected_sources:
        return None

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    actual = {
        normalize_source(source)
        for source in actual_sources
    }

    return 1.0 if expected.issubset(actual) else 0.0


# ============================================================
# UNANSWERABLE TEST
# ============================================================

def check_abstention(answer):

    """
    Used for questions where the answer does not exist
    in the document collection.
    """

    answer = answer.lower()

    abstention_phrases = [

        "couldn't find enough information",

        "could not find enough information",

        "not provided",

        "not available",

        "cannot find",

        "not mentioned",

        "don't have enough information",

        "do not have enough information",
    ]

    return any(
        phrase in answer
        for phrase in abstention_phrases
    )


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    print("\n")
    print("=" * 70)
    print("POTENS RAG END-TO-END EVALUATION")
    print("=" * 70)

    print(
        f"\nTotal evaluation questions: "
        f"{len(EVAL_DATASET)}"
    )

    print(
        f"Endpoint: {ASK_URL}\n"
    )

    # ========================================================
    # LOOP THROUGH DATASET
    # ========================================================

    for index, item in enumerate(
        EVAL_DATASET,
        start=1
    ):

        question = item["question"]

        expected_answer = item.get(
            "expected_answer",
            ""
        )

        expected_sources = item.get(
            "expected_sources",
            []
        )

        category = item.get(
            "category",
            "unknown"
        )

        print("=" * 70)

        print(
            f"[{index}/{len(EVAL_DATASET)}] "
            f"{item.get('id')}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {question}"
        )

        # ====================================================
        # CALL RAG
        # ====================================================

        start_time = time.perf_counter()

        response = call_rag(question)

        latency = (
            time.perf_counter()
            - start_time
        )

        # ====================================================
        # EXTRACT RESPONSE
        # ====================================================

        answer = response.get(
            "answer",
            ""
        )

        confidence = response.get(
            "confidence",
            0.0
        )

        citations = response.get(
            "citations",
            []
        )

        actual_sources = extract_sources(
            citations
        )

        contexts = extract_contexts(
            citations
        )

        # ====================================================
        # SOURCE METRICS
        # ====================================================

        c_recall = citation_recall(
            expected_sources,
            actual_sources
        )

        c_precision = citation_precision(
            expected_sources,
            actual_sources
        )

        exact_source_match = source_match(
            expected_sources,
            actual_sources
        )

        # ====================================================
        # UNANSWERABLE QUESTIONS
        # ====================================================

        abstained_correctly = None

        if category == "unanswerable":

            abstained_correctly = check_abstention(
                answer
            )

        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "id":
                item.get("id"),

            "category":
                category,

            "question":
                question,

            "expected_answer":
                expected_answer,

            "actual_answer":
                answer,

            "expected_sources":
                expected_sources,

            "actual_sources":
                actual_sources,

            "contexts":
                contexts,

            "citations":
                citations,

            "confidence":
                confidence,

            "citation_precision":
                c_precision,

            "citation_recall":
                c_recall,

            "exact_source_match":
                exact_source_match,

            "abstained_correctly":
                abstained_correctly,

            "latency_seconds":
                round(latency, 3),
        }

        if "error" in response:

            result["error"] = response[
                "error"
            ]

        results.append(result)

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        print("\nExpected Answer:")

        print(
            expected_answer
        )

        print("\nActual Answer:")

        print(
            answer
        )

        print("\nExpected Sources:")

        print(
            expected_sources
        )

        print("\nRetrieved Sources:")

        print(
            actual_sources
        )

        print(
            f"\nConfidence: "
            f"{confidence}"
        )

        print(
            f"Citation Precision: "
            f"{c_precision}"
        )

        print(
            f"Citation Recall: "
            f"{c_recall}"
        )

        print(
            f"Latency: "
            f"{latency:.2f}s"
        )

        if abstained_correctly is not None:

            print(
                f"Correct Abstention: "
                f"{abstained_correctly}"
            )

        print()

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    latencies = [
        result["latency_seconds"]
        for result in results
    ]

    citation_precision_scores = [

        result["citation_precision"]

        for result in results

        if result["citation_precision"]
        is not None
    ]

    citation_recall_scores = [

        result["citation_recall"]

        for result in results

        if result["citation_recall"]
        is not None
    ]

    source_matches = [

        result["exact_source_match"]

        for result in results

        if result["exact_source_match"]
        is not None
    ]

    abstention_scores = [

        result["abstained_correctly"]

        for result in results

        if result["abstained_correctly"]
        is not None
    ]

    # ========================================================
    # AVERAGES
    # ========================================================

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    avg_citation_precision = (
        sum(citation_precision_scores)
        / len(citation_precision_scores)
        if citation_precision_scores
        else 0
    )

    avg_citation_recall = (
        sum(citation_recall_scores)
        / len(citation_recall_scores)
        if citation_recall_scores
        else 0
    )

    avg_source_match = (
        sum(source_matches)
        / len(source_matches)
        if source_matches
        else 0
    )

    avg_abstention = (
        sum(abstention_scores)
        / len(abstention_scores)
        if abstention_scores
        else 0
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal Questions       : "
        f"{len(results)}"
    )

    print(
        f"Avg Citation Precision: "
        f"{avg_citation_precision:.2%}"
    )

    print(
        f"Avg Citation Recall   : "
        f"{avg_citation_recall:.2%}"
    )

    print(
        f"Exact Source Match    : "
        f"{avg_source_match:.2%}"
    )

    if abstention_scores:

        print(
            f"Correct Abstention    : "
            f"{avg_abstention:.2%}"
        )

    print(
        f"Average Latency       : "
        f"{avg_latency:.2f} sec"
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_evaluation()