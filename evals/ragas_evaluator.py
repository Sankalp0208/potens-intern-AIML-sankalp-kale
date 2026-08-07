"""
evals/ragas_evaluator.py

Potens RAG Project - RAGAS Evaluation
RAGAS Version: 0.4.3

Judge:
    Groq
    llama-3.3-70b-versatile

Embeddings:
    sentence-transformers/all-MiniLM-L6-v2

Metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall

Input:
    evals/results/evaluation_results.json

Output:
    evals/results/ragas_results.json
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from app.core.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = Path(
    "evals/results/evaluation_results.json"
)

RAGAS_RESULTS_FILE = Path(
    "evals/results/ragas_results.json"
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

def load_results() -> List[Dict[str, Any]]:

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            "\nEvaluation results file not found.\n\n"
            f"Expected:\n{RESULTS_FILE}\n\n"
            "Run first:\n"
            "python -m evals.run_evaluation"
        )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "evaluation_results.json must contain a JSON list."
        )

    return data


# ============================================================
# CREATE ASYNC GROQ CLIENT
# ============================================================

def create_evaluator_llm():
    """
    Create an asynchronous OpenAI-compatible client
    connected to Groq.

    RAGAS 0.4 Collections metrics call agenerate(),
    therefore an async client is required.
    """

    if not GROQ_API_KEY:

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )

    evaluator_llm = llm_factory(
        model=MODEL_NAME,

        # Important:
        # Groq exposes an OpenAI-compatible API.
        provider="openai",

        client=client,

        # Use Instructor structured output adapter.
        adapter="instructor",

        temperature=0,
    )

    return evaluator_llm


# ============================================================
# CREATE MODERN RAGAS EMBEDDINGS
# ============================================================

def create_evaluator_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        device="cpu",
    )

    return embeddings


# ============================================================
# CLEAN CONTEXTS
# ============================================================

def clean_contexts(
    contexts: Any,
) -> List[str]:

    if not contexts:

        return []

    cleaned = []

    for context in contexts:

        if context is None:
            continue

        text = str(context).strip()

        if text:

            cleaned.append(text)

    return cleaned


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_rows(
    results: List[Dict[str, Any]],
):

    valid_rows = []
    skipped_rows = []

    for index, item in enumerate(
        results,
        start=1,
    ):

        item_id = item.get(
            "id",
            f"Q{index}",
        )

        category = item.get(
            "category",
            "unknown",
        )

        question = str(
            item.get(
                "question",
                "",
            )
            or ""
        ).strip()

        answer = str(
            item.get(
                "actual_answer",
                "",
            )
            or ""
        ).strip()

        expected_answer = str(
            item.get(
                "expected_answer",
                "",
            )
            or ""
        ).strip()

        contexts = clean_contexts(
            item.get(
                "contexts",
                [],
            )
        )

        skip_reasons = []

        if not question:
            skip_reasons.append(
                "missing question"
            )

        if not answer:
            skip_reasons.append(
                "missing actual_answer"
            )

        if not expected_answer:
            skip_reasons.append(
                "missing expected_answer"
            )

        if not contexts:
            skip_reasons.append(
                "missing contexts"
            )

        if skip_reasons:

            skipped_rows.append(
                {
                    "id": item_id,
                    "category": category,
                    "reasons": skip_reasons,
                }
            )

            continue

        valid_rows.append(
            {
                "id": item_id,

                "category": category,

                "user_input": question,

                "response": answer,

                "reference": expected_answer,

                "retrieved_contexts": contexts,
            }
        )

    return valid_rows, skipped_rows


# ============================================================
# SCORE EXTRACTION
# ============================================================

def extract_score(
    result: Any,
) -> Optional[float]:

    if result is None:
        return None

    if hasattr(result, "value"):
        value = result.value
    else:
        value = result

    try:
        value = float(value)

    except (TypeError, ValueError):
        return None

    if math.isnan(value):
        return None

    return value


# ============================================================
# REASON EXTRACTION
# ============================================================

def extract_reason(
    result: Any,
) -> Optional[str]:

    if result is None:
        return None

    reason = getattr(
        result,
        "reason",
        None,
    )

    if reason is None:
        return None

    return str(reason)


# ============================================================
# RUN ONE METRIC SAFELY
# ============================================================

async def run_metric(
    metric_name: str,
    metric: Any,
    **kwargs,
) -> Dict[str, Any]:

    try:

        result = await metric.ascore(
            **kwargs
        )

        return {
            "score": extract_score(result),
            "reason": extract_reason(result),
            "error": None,
        }

    except Exception as error:

        error_message = (
            f"{type(error).__name__}: {error}"
        )

        print(
            f"\n    [WARNING] {metric_name} failed"
        )

        print(
            f"    {error_message}"
        )

        return {
            "score": None,
            "reason": None,
            "error": error_message,
        }


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

async def evaluate_row(
    row: Dict[str, Any],
    faithfulness_metric: Any,
    answer_relevancy_metric: Any,
    context_precision_metric: Any,
    context_recall_metric: Any,
) -> Dict[str, Any]:

    question = row["user_input"]

    answer = row["response"]

    reference = row["reference"]

    contexts = row["retrieved_contexts"]


    # --------------------------------------------------------
    # FAITHFULNESS
    # --------------------------------------------------------

    faithfulness_result = await run_metric(
        "Faithfulness",

        faithfulness_metric,

        user_input=question,

        response=answer,

        retrieved_contexts=contexts,
    )


    # --------------------------------------------------------
    # ANSWER RELEVANCY
    # --------------------------------------------------------

    answer_relevancy_result = await run_metric(
        "Answer Relevancy",

        answer_relevancy_metric,

        user_input=question,

        response=answer,
    )


    # --------------------------------------------------------
    # CONTEXT PRECISION
    # --------------------------------------------------------

    context_precision_result = await run_metric(
        "Context Precision",

        context_precision_metric,

        user_input=question,

        reference=reference,

        retrieved_contexts=contexts,
    )


    # --------------------------------------------------------
    # CONTEXT RECALL
    # --------------------------------------------------------

    context_recall_result = await run_metric(
        "Context Recall",

        context_recall_metric,

        user_input=question,

        reference=reference,

        retrieved_contexts=contexts,
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "id":
            row["id"],

        "category":
            row["category"],

        "question":
            question,

        "expected_answer":
            reference,

        "actual_answer":
            answer,

        "retrieved_contexts":
            contexts,


        "faithfulness":
            faithfulness_result["score"],

        "answer_relevancy":
            answer_relevancy_result["score"],

        "context_precision":
            context_precision_result["score"],

        "context_recall":
            context_recall_result["score"],


        "reasons": {

            "faithfulness":
                faithfulness_result["reason"],

            "answer_relevancy":
                answer_relevancy_result["reason"],

            "context_precision":
                context_precision_result["reason"],

            "context_recall":
                context_recall_result["reason"],
        },


        "errors": {

            "faithfulness":
                faithfulness_result["error"],

            "answer_relevancy":
                answer_relevancy_result["error"],

            "context_precision":
                context_precision_result["error"],

            "context_recall":
                context_recall_result["error"],
        },
    }


# ============================================================
# CALCULATE AVERAGE
# ============================================================

def calculate_average(
    results: List[Dict[str, Any]],
    metric_name: str,
) -> Optional[float]:

    scores = [
        row[metric_name]

        for row in results

        if row.get(metric_name) is not None
    ]

    if not scores:
        return None

    return sum(scores) / len(scores)


# ============================================================
# DISPLAY SCORE
# ============================================================

def display_score(
    score: Optional[float],
) -> str:

    if score is None:
        return "FAILED"

    return (
        f"{score:.4f} "
        f"({score * 100:.2f}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    raw_results,
    evaluated_results,
    skipped_rows,
    averages,
    successful_counts,
    failures,
):

    output = {

        "configuration": {

            "ragas_version":
                "0.4.3",

            "llm_provider":
                "Groq OpenAI-compatible API",

            "llm_model":
                MODEL_NAME,

            "embedding_provider":
                "RAGAS HuggingFaceEmbeddings",

            "embedding_model":
                EMBEDDING_MODEL,
        },


        "summary": {

            "total_questions":
                len(raw_results),

            "ragas_evaluated":
                len(evaluated_results),

            "ragas_skipped":
                len(skipped_rows),

            "metrics":
                averages,

            "successful_metric_evaluations":
                successful_counts,

            "metric_failures":
                failures,
        },


        "skipped_rows":
            skipped_rows,


        "results":
            evaluated_results,
    }


    RAGAS_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        RAGAS_RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# MAIN
# ============================================================

async def run_ragas_evaluation():

    print("\n")
    print("=" * 70)
    print("POTENS RAGAS EVALUATION")
    print("RAGAS VERSION : 0.4.3")
    print("=" * 70)


    # ========================================================
    # LOAD RESULTS
    # ========================================================

    raw_results = load_results()

    print(
        f"\nLoaded evaluation results : "
        f"{len(raw_results)}"
    )


    # ========================================================
    # PREPARE ROWS
    # ========================================================

    rows, skipped_rows = prepare_rows(
        raw_results
    )

    print(
        f"RAGAS rows prepared       : "
        f"{len(rows)}"
    )

    print(
        f"Rows skipped              : "
        f"{len(skipped_rows)}"
    )


    if not rows:

        print(
            "\nNo valid rows available."
        )

        return


    # ========================================================
    # DISPLAY CONFIG
    # ========================================================

    print(
        f"\nEvaluator LLM             : "
        f"{MODEL_NAME}"
    )

    print(
        "LLM Provider              : "
        "Groq OpenAI-compatible API"
    )

    print(
        "Client                    : "
        "AsyncOpenAI"
    )

    print(
        f"Embedding Model           : "
        f"{EMBEDDING_MODEL}"
    )


    # ========================================================
    # LLM
    # ========================================================

    print(
        "\nInitializing async Groq RAGAS judge..."
    )

    evaluator_llm = create_evaluator_llm()

    print(
        "Async Groq RAGAS judge initialized."
    )


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    print(
        "\nInitializing RAGAS HuggingFace embeddings..."
    )

    evaluator_embeddings = (
        create_evaluator_embeddings()
    )

    print(
        "RAGAS HuggingFace embeddings initialized."
    )


    # ========================================================
    # METRICS
    # ========================================================

    print(
        "\nInitializing RAGAS metrics..."
    )


    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )


    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )


    context_precision_metric = ContextPrecision(
        llm=evaluator_llm
    )


    context_recall_metric = ContextRecall(
        llm=evaluator_llm
    )


    print(
        "RAGAS metrics initialized."
    )


    print("\nMetrics:")

    print(
        "  1. Faithfulness"
    )

    print(
        "  2. Answer Relevancy"
    )

    print(
        "  3. Context Precision"
    )

    print(
        "  4. Context Recall"
    )


    # ========================================================
    # START
    # ========================================================

    print(
        "\nStarting RAGAS evaluation..."
    )

    print(
        f"Questions: {len(rows)}"
    )

    print(
        "Multiple Groq API calls "
        "will be made per question.\n"
    )


    evaluated_results = []

    total = len(rows)


    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for index, row in enumerate(
        rows,
        start=1,
    ):

        print(
            "=" * 70
        )

        print(
            f"[{index}/{total}] "
            f"{row['id']} "
            f"({row['category']})"
        )

        print(
            f"Question: "
            f"{row['user_input']}"
        )


        result = await evaluate_row(

            row,

            faithfulness_metric,

            answer_relevancy_metric,

            context_precision_metric,

            context_recall_metric,
        )


        evaluated_results.append(
            result
        )


        print()

        print(
            "    Faithfulness      : "
            + display_score(
                result["faithfulness"]
            )
        )

        print(
            "    Answer Relevancy  : "
            + display_score(
                result["answer_relevancy"]
            )
        )

        print(
            "    Context Precision : "
            + display_score(
                result["context_precision"]
            )
        )

        print(
            "    Context Recall    : "
            + display_score(
                result["context_recall"]
            )
        )

        print()


    # ========================================================
    # SUMMARY
    # ========================================================

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]


    display_names = {

        "faithfulness":
            "Faithfulness",

        "answer_relevancy":
            "Answer Relevancy",

        "context_precision":
            "Context Precision",

        "context_recall":
            "Context Recall",
    }


    averages = {

        metric:
            calculate_average(
                evaluated_results,
                metric,
            )

        for metric in metric_names
    }


    successful_counts = {

        metric:

            sum(
                1

                for row in evaluated_results

                if row.get(metric) is not None
            )

        for metric in metric_names
    }


    failures = {

        metric:

            len(evaluated_results)
            -
            successful_counts[metric]

        for metric in metric_names
    }


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 70)


    print(
        f"\nTotal Questions       : "
        f"{len(raw_results)}"
    )

    print(
        f"RAGAS Evaluated       : "
        f"{len(evaluated_results)}"
    )

    print(
        f"RAGAS Skipped         : "
        f"{len(skipped_rows)}"
    )


    print(
        "\nAverage RAGAS Scores:\n"
    )


    for metric in metric_names:

        print(
            f"{display_names[metric]:20} : "
            f"{display_score(averages[metric])}"
        )


    print(
        "\nSuccessful Evaluations:\n"
    )


    for metric in metric_names:

        print(
            f"{display_names[metric]:20} : "
            f"{successful_counts[metric]}"
            f"/{len(evaluated_results)}"
        )


    print(
        "\nMetric Failures:\n"
    )


    for metric in metric_names:

        print(
            f"{display_names[metric]:20} : "
            f"{failures[metric]}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        raw_results,
        evaluated_results,
        skipped_rows,
        averages,
        successful_counts,
        failures,
    )


    print("\n")
    print("=" * 70)
    print("RAGAS EVALUATION COMPLETE")
    print("=" * 70)


    print(
        f"\nResults saved to:\n"
        f"{RAGAS_RESULTS_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        run_ragas_evaluation()
    )