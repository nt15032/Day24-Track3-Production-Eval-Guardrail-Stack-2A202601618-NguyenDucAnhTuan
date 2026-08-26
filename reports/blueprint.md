# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Đức Anh Tuấn

**Ngày:** 26/08/2026

## Guard Stack Architecture

```text
User Input
    │
    ▼
[Presidio-compatible PII Scan] -- CCCD / CMND / phone / email
    │ block: HTTP 400 + redacted audit event
    ▼
[Local Rules + NeMo Input Rail]
    │ block: jailbreak / prompt injection / PII request / off-topic
    ▼
[Day 18 RAG Pipeline]
    │ hierarchical chunks → BM25+dense RRF → cross-encoder → answer
    ▼
[PII/Sensitive Output Check + NeMo Output Rail]
    │ block/redact unsafe output
    ▼
User Response
```

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| PII detection fallback | 0.014 | 0.028 | 0.058 | <10 ms |
| Local input policy | 0.008 | 0.023 | 0.025 | <300 ms |
| RAG Pipeline | N/A | N/A | N/A | <2000 ms |
| Hosted NeMo output rail | N/A | N/A | N/A | <300 ms |
| **Total measured local guard** | **0.021** | **0.038** | **0.083** | **<500 ms** |

**Measured local budget:** Pass.

**Production status:** Chưa xác nhận. Cần benchmark lại với Presidio spaCy, NeMo và API model thật; số đo local không đại diện cho network latency.

## CI/CD Gates

```yaml
- name: Unit and guard tests
  run: pytest tests/ -v

- name: RAGAS quality gate
  run: python src/phase_a_ragas.py
  policy:
    min_faithfulness: 0.75
    min_average_score: 0.65
    require_hosted_ragas_for_release: true

- name: Adversarial guard gate
  run: python src/phase_c_guard.py
  policy:
    minimum_pass_rate: 0.90

- name: Guard latency gate
  policy:
    maximum_total_p95_ms: 500
```

Release build phải dùng hosted RAGAS/NeMo, không chấp nhận `offline_proxy` làm bằng chứng production. Pull request có thể dùng fallback để kiểm tra logic nhanh; nightly/release workflow chạy dependency và API integration đầy đủ.

## Monitoring Dashboard

| Metric | Alert Threshold | Action |
|---|---:|---|
| RAGAS faithfulness, daily sample | <0.70 | Page on-call và giữ bản deploy trước |
| Adversarial suite pass rate | <90% | Block release, cập nhật attack patterns |
| Guard P95 latency | >500 ms | Kiểm tra model/API, timeout và circuit breaker |
| PII detections | >10/hour hoặc tăng 3× baseline | Security alert, kiểm tra nguồn traffic |
| NeMo unavailable rate | >1% trong 5 phút | Fail closed cho dữ liệu nhạy cảm, bật fallback |
| Version-conflict failures | >5% | Kiểm tra metadata hiệu lực và retrieval filter |

## Kết quả thực tế từ Lab

| Metric | Kết quả |
|---|---:|
| RAGAS/proxy avg_score (50q) | 0.8344 |
| Worst aggregate metric | answer_relevancy (0.7133) |
| Dominant failure distribution | adversarial (avg 0.7821) |
| Cohen's κ | 0.4000 |
| Adversarial pass rate | 20/20 (100%) |
| Local guard P95 latency | 0.038 ms |

## Nhận xét & Cải tiến

Guard local xử lý đúng toàn bộ 20 mẫu tấn công và có latency rất thấp. Retrieval factual tốt hơn multi-hop/adversarial; điểm yếu chính là tổng hợp câu trả lời và recall khi phải kết hợp hoặc phân giải nhiều policy. Trước production cần cài Qdrant, Presidio, NeMo, model embedding/reranker và chạy lại bằng API thật. Cũng cần thêm metadata `status`, `effective_date`, `supersedes` để ưu tiên policy hiện hành, cùng circuit breaker và fail-closed behavior khi guard service lỗi.
