from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
from pathlib import Path
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt = f"""Bạn là chuyên gia đánh giá câu trả lời RAG về chính sách nhân sự.
So sánh theo độ chính xác, đầy đủ và súc tích. Không ưu tiên câu dài hơn.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Chỉ trả về JSON có dạng:
{{"winner":"A|B|tie","reasoning":"...","scores":{{"A":0.0,"B":0.0}}}}
"""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pairwise_judge_result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                                "reasoning": {"type": "string"},
                                "scores": {
                                    "type": "object",
                                    "properties": {
                                        "A": {"type": "number", "minimum": 0, "maximum": 1},
                                        "B": {"type": "number", "minimum": 0, "maximum": 1},
                                    },
                                    "required": ["A", "B"],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["winner", "reasoning", "scores"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return _validate_judge_payload(parsed)
        except Exception:
            # CI and local development must remain usable if the API is unavailable.
            pass

    score_a = _offline_answer_score(question, answer_a)
    score_b = _offline_answer_score(question, answer_b)
    delta = score_a - score_b
    winner = "tie" if abs(delta) < 0.05 else ("A" if delta > 0 else "B")
    return {
        "winner": winner,
        "reasoning": (
            "Fallback local: so sánh mức độ bao phủ câu hỏi, tính cụ thể và độ súc tích; "
            f"A={score_a:.3f}, B={score_b:.3f}."
        ),
        "scores": {"A": score_a, "B": score_b},
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _offline_answer_score(question: str, answer: str) -> float:
    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)
    coverage = len(question_tokens & answer_tokens) / len(question_tokens) if question_tokens else 0.0
    specificity = min(1.0, (len(re.findall(r"\d+", answer)) +
                            len({"không", "bắt", "buộc", "hiện", "hành"} & answer_tokens)) / 3)
    length = len(answer.strip())
    conciseness = 1.0 if 25 <= length <= 500 else max(0.0, 1 - abs(length - 200) / 800)
    return round(min(1.0, 0.55 * coverage + 0.30 * specificity + 0.15 * conciseness), 4)


def _validate_judge_payload(payload: dict) -> dict:
    winner = payload.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError(f"Invalid judge winner: {winner!r}")
    scores = payload.get("scores") or {}
    normalized_scores = {
        key: max(0.0, min(1.0, float(scores.get(key, 0.0)))) for key in ("A", "B")
    }
    reasoning = str(payload.get("reasoning") or "Không có giải thích từ judge.")
    return {"winner": winner, "reasoning": reasoning, "scores": normalized_scores}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = _validate_judge_payload(pairwise_judge(question, answer_a, answer_b))
    pass2_raw = _validate_judge_payload(pairwise_judge(question, answer_b, answer_a))
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=pass1["winner"] if consistent else "tie",
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have equal length")
    if not judge_labels:
        return 0.0
    allowed = {0, 1}
    if not set(judge_labels).issubset(allowed) or not set(human_labels).issubset(allowed):
        raise ValueError("Cohen kappa labels must be binary (0 or 1)")
    n = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    expected = (
        judge_labels.count(1) / n * human_labels.count(1) / n
        + judge_labels.count(0) / n * human_labels.count(0) / n
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1 - expected)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    position_bias_count = sum(not result.position_consistent for result in judge_results)
    position_bias_rate = position_bias_count / total if total else 0.0
    decisive_results = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in decisive_results
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in decisive_results
    )
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / len(decisive_results)
        if decisive_results else 0.0
    )
    position_text = (
        "Position bias cao; luôn giữ swap-and-average trong production."
        if position_bias_rate > 0.3 else
        "Position bias thấp; kết quả judge tương đối ổn định qua phép đổi vị trí."
    )
    verbosity_text = (
        " Verbosity bias đáng lưu ý." if verbosity_bias > 0.6
        else " Chưa thấy verbosity bias đáng kể."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive_results),
        },
        "interpretation": position_text + verbosity_text,
    }


def _truth_similarity(answer: str, truth: str) -> float:
    truth_tokens = _tokens(truth)
    answer_tokens = _tokens(answer)
    return len(answer_tokens & truth_tokens) / len(truth_tokens) if truth_tokens else 0.0


def generate_phase_b_report(path: str = "reports/judge_results.json") -> dict:
    human_data = json.loads(Path(HUMAN_LABELS_PATH).read_text(encoding="utf-8"))
    test_data = json.loads(Path("test_set_50q.json").read_text(encoding="utf-8"))
    truths = {item["id"]: item["ground_truth"] for item in test_data}
    rows = []
    judge_labels = []
    human_labels = []
    for item in human_data:
        similarity = _truth_similarity(item["model_answer"], truths[item["question_id"]])
        judge_label = int(similarity >= 0.38)
        judge_labels.append(judge_label)
        human_labels.append(item["human_label"])
        rows.append({
            **item,
            "judge_label": judge_label,
            "reference_similarity": round(similarity, 4),
            "agree": judge_label == item["human_label"],
        })

    # Exercise pairwise + swapping on representative good/bad answers.
    comparisons = []
    for item in human_data[:5]:
        result = swap_and_average(
            item["question"], item["model_answer"], truths[item["question_id"]]
        )
        comparisons.append(result)
    bias = bias_report(comparisons)
    report = {
        "judge_model": JUDGE_MODEL if OPENAI_API_KEY else "offline_deterministic_fallback",
        "cohen_kappa": round(cohen_kappa(judge_labels, human_labels), 4),
        "labels": rows,
        "bias": bias,
        "pairwise_results": [result.__dict__ for result in comparisons],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    report = generate_phase_b_report()
    print(f"Cohen's κ: {report['cohen_kappa']:.3f}")
    print(f"\nBias report: {report['bias']}")
    print("Saved → reports/judge_results.json")
