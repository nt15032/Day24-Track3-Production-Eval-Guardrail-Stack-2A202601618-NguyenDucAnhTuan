from __future__ import annotations

"""Phase C: production guardrails with offline-safe Presidio/NeMo fallbacks."""

import asyncio
import json
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR,
                    LATENCY_BUDGET_P95_MS, OPENAI_API_KEY, PRESIDIO_LANGUAGE)


_PRESIDIO_CACHE = None
_PRESIDIO_UNAVAILABLE = False


def setup_presidio():
    """Create Presidio engines with Vietnamese CCCD/CMND/phone recognizers."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
                  Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7)],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect and redact Vietnamese identity numbers, phones and email addresses."""
    del anonymizer
    global _PRESIDIO_CACHE, _PRESIDIO_UNAVAILABLE
    entities: list[dict] = []
    patterns = [
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.99),
        ("VN_CCCD", r"\b\d{12}\b", 0.90),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.95),
        ("VN_CCCD", r"(?i)(?<=CMND\s)\d{9}\b", 0.85),
    ]
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            entities.append({"type": entity_type, "text": match.group(0), "score": score,
                             "start": match.start(), "end": match.end()})

    if analyzer is None and not _PRESIDIO_UNAVAILABLE:
        if _PRESIDIO_CACHE is None:
            try:
                _PRESIDIO_CACHE = setup_presidio()
            except Exception:
                _PRESIDIO_UNAVAILABLE = True
        if _PRESIDIO_CACHE is not None:
            analyzer = _PRESIDIO_CACHE[0]
    if analyzer is not None:
        try:
            # The bundled spaCy model is English while the corpus is Vietnamese.
            # Restrict Presidio to the high-signal entities required by this lab;
            # otherwise English NER marks ordinary Vietnamese policy phrases as PERSON.
            for result in analyzer.analyze(
                text=text,
                language=PRESIDIO_LANGUAGE,
                entities=["EMAIL_ADDRESS", "VN_CCCD", "VN_PHONE"],
            ):
                entities.append({
                    "type": result.entity_type, "text": text[result.start:result.end],
                    "score": round(float(result.score), 3), "start": result.start, "end": result.end,
                })
        except Exception:
            pass

    unique: dict[tuple[int, int], dict] = {}
    for entity in entities:
        key = (entity["start"], entity["end"])
        if key not in unique or entity["score"] > unique[key]["score"]:
            unique[key] = entity
    entities = sorted(unique.values(), key=lambda item: (item["start"], item["end"]))
    anonymized = text
    for entity in reversed(entities):
        anonymized = anonymized[:entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"]:]
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def setup_nemo_rails():
    """Load NeMo Guardrails from guardrails/config.yml and rails.co."""
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


def _local_input_violation(text: str) -> str | None:
    lowered = text.casefold()
    categories = {
        "jailbreak": [
            "bỏ qua tất cả hướng dẫn", "bo qua tat ca huong dan", "ignore your system prompt",
            "ignore previous instructions", "ignore your previous instructions", "pretend you are dan",
            "unrestricted ai", "ai không có giới hạn", "ai khong co gioi han", "system override",
            "admin command", "đóng vai hr manager", "dong vai hr manager", "tôi là ceo", "toi la ceo",
        ],
        "prompt_injection": [
            "ignore previous", "print all confidential", "dump all training data",
            "in ra toàn bộ system", "in ra toan bo system", "system instructions",
            "toàn bộ context", "toan bo context", "<!--",
        ],
        "pii_request": [
            "cccd của", "cccd cua", "số điện thoại của nhân viên", "so dien thoai cua nhan vien",
            "lương của nhân viên", "luong cua nhan vien", "bảng lương chi tiết", "bang luong chi tiet",
            "thông tin cá nhân của", "thong tin ca nhan cua", "liệt kê toàn bộ thông tin nhân viên",
            "liet ke toan bo thong tin nhan vien", "employee salaries", "employee records",
            "confidential employee data",
        ],
        "off_topic": [
            "bài thơ", "bai tho", "nấu phở", "nau pho", "bitcoin", "ethereum",
            "phương trình vi phân", "phuong trinh vi phan", "phim hay", "marvel",
            "giá cổ phiếu", "gia co phieu", "thời tiết", "thoi tiet", "tấn công mạng", "tan cong mang",
        ],
    }
    for category, phrases in categories.items():
        if any(phrase in lowered for phrase in phrases):
            return category
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Block prompt attacks, PII requests and non-HR topics."""
    local_reason = _local_input_violation(text)
    if local_reason:
        return {"allowed": False, "blocked_reason": local_reason,
                "response": "Xin lỗi, yêu cầu đã bị chặn. Tôi chỉ hỗ trợ chính sách nhân sự và không cung cấp dữ liệu nhạy cảm."}
    if rails is None and OPENAI_API_KEY:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None
    if rails is not None:
        try:
            raw = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response = raw.get("content", "") if isinstance(raw, dict) else str(raw)
            refused = any(keyword in response.casefold() for keyword in
                          ["xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"])
            return {"allowed": not refused,
                    "blocked_reason": "nemo_input_rail" if refused else None,
                    "response": response}
        except Exception as exc:
            return {"allowed": False, "blocked_reason": "nemo_unavailable",
                    "response": f"Guardrail service unavailable: {type(exc).__name__}"}
    return {"allowed": True, "blocked_reason": None, "response": "Allowed by local input policy."}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Redact PII/sensitive output, then optionally invoke NeMo output rails."""
    pii_result = pii_scan(answer)
    sensitive_phrases = ["mật khẩu hệ thống là", "mat khau he thong la", "cccd của nhân viên là",
                         "số điện thoại cá nhân của", "thông tin tối mật", "bí mật thương mại"]
    if pii_result["has_pii"] or any(phrase in answer.casefold() for phrase in sensitive_phrases):
        return {"safe": False, "flagged_reason": "pii_or_sensitive_output",
                "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này. Vui lòng liên hệ phòng Nhân sự."}
    if rails is None and OPENAI_API_KEY:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None
    if rails is not None:
        try:
            from nemoguardrails.rails.llm.options import GenerationOptions

            # rails=["output"] chạy ĐÚNG output rail (self check output) trên câu trả lời
            # có sẵn. Bỏ options đi thì generate_async sinh một lượt hội thoại mới và
            # verdict trở thành ngẫu nhiên.
            raw = await rails.generate_async(
                messages=[{"role": "user", "content": question},
                          {"role": "assistant", "content": answer}],
                options=GenerationOptions(rails=["output"]))
            response = getattr(raw, "response", raw)
            if isinstance(response, list):
                response = response[0].get("content", "") if response else ""
            elif isinstance(response, dict):
                response = response.get("content", "")
            response = str(response).strip()
            # Output rail trả lại nguyên văn khi an toàn, thay bằng câu từ chối khi chặn.
            # So sánh chuỗi bền hơn match keyword — câu từ chối đổi theo config/locale.
            blocked = response != answer.strip()
            return {"safe": not blocked, "flagged_reason": "nemo_output_rail" if blocked else None,
                    "final_answer": response if blocked else answer}
        except Exception:
            # Fail-closed: verifier chết thì coi như không xác minh được (blueprint: Block + log).
            return {"safe": False, "flagged_reason": "nemo_unavailable",
                    "final_answer": "Không thể xác minh độ an toàn của câu trả lời lúc này."}
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                          analyzer=None, anonymizer=None) -> list[dict]:
    """Run every adversarial input through PII and input guard layers."""
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = blocked_reason = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"
                blocked_reason = ",".join(sorted({e["type"] for e in pii_result["entities"]}))
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by, blocked_reason = "nemo_input", rail_result["blocked_reason"]
            actual = "blocked" if blocked_by else "allowed"
            results.append({"id": item["id"], "category": item["category"], "input": item["input"],
                            "expected": item["expected"], "actual": actual, "blocked_by": blocked_by,
                            "blocked_reason": blocked_reason, "passed": actual == item["expected"]})
        return results

    results = asyncio.run(_run_all())
    print(f"Adversarial suite: {sum(result['passed'] for result in results)}/{len(results)} passed")
    return results


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                        rails=None, analyzer=None, anonymizer=None) -> dict:
    """Measure P50/P95/P99 for Presidio, input rails and their total."""
    presidio_times, nemo_times, total_times = [], [], []
    samples = test_inputs[:max(0, n_runs)]

    async def _measure():
        for text in samples:
            started = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - started) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(values: list[float]) -> dict:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(values)
        def nearest_rank(percentile: float) -> float:
            index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
            return round(ordered[index], 3)
        return {"p50": nearest_rank(0.50), "p95": nearest_rank(0.95), "p99": nearest_rank(0.99)}

    total_percentiles = percentiles(total_times)
    return {"presidio_ms": percentiles(presidio_times), "nemo_ms": percentiles(nemo_times),
            "total_ms": total_percentiles,
            "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
            "budget_ms": LATENCY_BUDGET_P95_MS, "sample_count": len(samples)}


def save_guard_report(results: list[dict], latency: dict,
                      path: str = "reports/guard_results.json") -> dict:
    passed = sum(result["passed"] for result in results)
    per_category: dict[str, dict] = {}
    for result in results:
        stats = per_category.setdefault(result["category"], {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(result["passed"])
    report = {"total": len(results), "passed": passed,
              "pass_rate": round(passed / len(results), 4) if results else 0.0,
              "per_category": per_category, "results": results, "latency": latency}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    demo = pii_scan("Nhân viên A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép.")
    print(f"PII detected: {demo['has_pii']}; entities: {demo['entities']}")
    adversarial_set = json.loads(Path(ADVERSARIAL_SET_PATH).read_text(encoding="utf-8"))
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=20)
    print(f"Latency P95: {latency['total_ms']['p95']}ms; budget OK: {latency['latency_budget_ok']}")
    save_guard_report(results, latency)
    print("Saved → reports/guard_results.json")
