from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re, unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, EMBEDDING_MODEL


def _safe_float(v) -> float:
    """RAGAS returns NaN for metrics that errored on a row."""
    try:
        v = float(v)
        return 0.0 if math.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    stop_words = {"la", "va", "cua", "co", "duoc", "cho", "mot", "nhan", "vien",
                  "theo", "trong", "khi", "bao", "nhieu", "nao", "thi", "gi"}
    return {token for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 1 and token not in stop_words}


def _coverage(source: str, target: str) -> float:
    source_tokens, target_tokens = _tokens(source), _tokens(target)
    return len(source_tokens & target_tokens) / len(target_tokens) if target_tokens else 0.0


def _fallback_evaluation(questions: list[str], answers: list[str],
                         contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Deterministic proxy metrics for offline CI; hosted RAGAS remains preferred."""
    rows = []
    for question, answer, context_list, truth in zip(questions, answers, contexts, ground_truths):
        context_text = "\n".join(context_list)
        relevance = max(_coverage(answer, question), _coverage(answer, truth))
        relevant_contexts = sum(_coverage(context, truth) >= 0.15 for context in context_list)
        rows.append(EvalResult(
            question=question, answer=answer, contexts=context_list, ground_truth=truth,
            faithfulness=round(min(1.0, _coverage(context_text, answer)), 4),
            answer_relevancy=round(min(1.0, relevance), 4),
            context_precision=round(relevant_contexts / len(context_list), 4) if context_list else 0.0,
            context_recall=round(min(1.0, _coverage(context_text, truth)), 4),
        ))

    def average(name: str) -> float:
        return round(sum(getattr(row, name) for row in rows) / len(rows), 4) if rows else 0.0

    return {"faithfulness": average("faithfulness"),
            "answer_relevancy": average("answer_relevancy"),
            "context_precision": average("context_precision"),
            "context_recall": average("context_recall"),
            "per_question": rows, "evaluator": "deterministic_token_overlap"}


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation. LLM judge = DeepSeek (OpenAI-compatible), embeddings = local bge-m3
    (DeepSeek has no embeddings endpoint, RAGAS needs one for answer_relevancy)."""
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal length")
    if not DEEPSEEK_API_KEY:
        return _fallback_evaluation(questions, answers, contexts, ground_truths)
    # datasets/huggingface_hub sometimes gate a metric load behind an interactive
    # "this repo has custom code, run it? [y/N]" input() prompt — with no TTY
    # attached (subprocess run from Colab) that silently hangs forever instead
    # of raising. Trust it non-interactively and belt-and-suspenders auto-answer
    # any input() call so evaluate() can never block on stdin.
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    import builtins
    _original_input = builtins.input

    def _auto_yes_input(prompt=""):
        print(f"[evaluate_ragas] auto-answered 'y' to input() prompt: {prompt!r}")
        return "y"

    builtins.input = _auto_yes_input
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_core.embeddings import Embeddings

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        llm = ChatOpenAI(model=LLM_MODEL, api_key=DEEPSEEK_API_KEY,
                          base_url=DEEPSEEK_BASE_URL, temperature=0)
        # DeepSeek's API only supports n=1; ragas' answer_relevancy defaults to
        # strictness=3 (asks the LLM for 3 completions in one call via n=3).
        answer_relevancy.strictness = 1

        class _SharedEncoderEmbeddings(Embeddings):
            """Wraps config.get_encoder() instead of loading bge-m3 a 2nd time —
            a fresh HuggingFaceEmbeddings() here reloads the same safetensors file
            DenseSearch already loaded, which access-violates on Windows."""
            def __init__(self, model_name):
                from config import get_encoder
                self._model = get_encoder(model_name)

            def embed_documents(self, texts):
                return self._model.encode(texts).tolist()

            def embed_query(self, text):
                return self._model.encode(text).tolist()

        embeddings = _SharedEncoderEmbeddings(EMBEDDING_MODEL)

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                           llm=llm, embeddings=embeddings, raise_exceptions=False)
        df = result.to_pandas()
        per_question = [EvalResult(question=row["question"], answer=row["answer"],
            contexts=row["contexts"], ground_truth=row["ground_truth"],
            faithfulness=_safe_float(row.get("faithfulness")),
            answer_relevancy=_safe_float(row.get("answer_relevancy")),
            context_precision=_safe_float(row.get("context_precision")),
            context_recall=_safe_float(row.get("context_recall")))
            for _, row in df.iterrows()]
        return {"faithfulness": _safe_float(df["faithfulness"].mean()),
                "answer_relevancy": _safe_float(df["answer_relevancy"].mean()),
                "context_precision": _safe_float(df["context_precision"].mean()),
                "context_recall": _safe_float(df["context_recall"].mean()),
                "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return _fallback_evaluation(questions, answers, contexts, ground_truths)
    finally:
        builtins.input = _original_input


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question, "worst_metric": worst_metric, "score": metrics[worst_metric],
            "avg_score": avg, "diagnosis": diagnosis, "suggested_fix": fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
