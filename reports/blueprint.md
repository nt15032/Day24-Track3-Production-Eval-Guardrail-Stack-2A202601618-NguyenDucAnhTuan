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

Đo trên Colab A100, Presidio + spaCy `en_core_web_lg` thật, NeMo Guardrails gọi `gpt-4o-mini`
qua network, 20 mẫu adversarial (`sample_count: 20`).

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Kết quả |
|---|---:|---:|---:|---:|:--|
| Presidio PII scan | 9.723 | 11.806 | 11.820 | <10 ms | ✗ vượt 1.8 ms ở P95 |
| NeMo input rail | 0.016 | 161.538 | 845.827 | <300 ms | ✓ P95 đạt, ✗ P99 vượt |
| RAG Pipeline | N/A | N/A | N/A | <2000 ms | chưa đo trong suite này |
| **Total guard stack** | **10.264** | **172.620** | **854.840** | **<500 ms** | **✓ Pass** |

**Ngân sách tổng:** Pass — P95 172.6 ms, còn dư 65% budget 500 ms.

**Hai điểm không đạt, không che giấu:**

- **Presidio P95 11.8 ms > 10 ms.** Lần đo local trước cho 0.028 ms vì Presidio khi đó không nạp được
  và `pii_scan()` lặng lẽ rơi về regex thuần. Con số 11.8 ms mới là chi phí thật của spaCy NER.
  Muốn về dưới 10 ms thì bỏ `en_core_web_lg` và chỉ giữ `PatternRecognizer` cho CCCD/phone/email —
  đánh đổi bằng việc mất khả năng bắt PERSON/LOCATION.
- **NeMo P99 845.8 ms.** P50 chỉ 0.016 ms vì phần lớn input bị chặn ngay ở lớp rule local, không
  chạm tới network; những request thật sự đi tới `gpt-4o-mini` mới tạo ra đuôi dài này. Production
  cần timeout cứng ~800 ms + circuit breaker, nếu không P99 sẽ kéo theo tail latency của cả API.

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

Toàn bộ số dưới đây đo bằng API thật (`evaluation_mode: ragas_deepseek`, `judge_model: gpt-4o-mini`,
Presidio + NeMo sống), không phải fallback offline.

| Metric | Kết quả |
|---|---:|
| RAGAS avg_score (50q) | 0.8392 |
| Worst aggregate metric | answer_relevancy (0.7227) |
| Dominant failure distribution | adversarial (avg 0.7272) |
| Cohen's κ | 0.4444 |
| Adversarial pass rate | 20/20 (100%) |
| Guard P95 latency | 172.620 ms |

### RAGAS theo distribution

| Distribution | n | faithfulness | answer_relevancy | context_precision | context_recall | avg |
|---|---:|---:|---:|---:|---:|---:|
| factual | 20 | 0.9629 | 0.8879 | 1.0000 | 0.9544 | **0.9513** |
| multi_hop | 20 | 0.7326 | 0.6615 | 0.9833 | 0.7545 | **0.7830** |
| adversarial | 10 | 0.8179 | 0.5148 | 0.9333 | 0.6429 | **0.7272** |

Thứ tự `factual > multi_hop > adversarial` đúng như thiết kế test set: pipeline yếu dần khi phải
kết hợp nhiều tài liệu, và yếu nhất khi gặp bẫy version conflict.

`context_precision` gần như hoàn hảo (0.98) trong khi `context_recall` chỉ 0.81 và `answer_relevancy`
0.72 — reranker lọc rất sạch nhưng **lọc quá tay**: những gì lấy về đều đúng, chỉ là chưa lấy đủ.
Với `RERANK_TOP_K = 3`, câu multi-hop cần 4–5 đoạn sẽ bị cắt mất phần cần thiết. Hướng sửa là nâng
top-k theo loại câu hỏi thay vì cố định 3.

## Nhận xét & Cải tiến

Guard stack chặn đúng toàn bộ 20 mẫu tấn công ở cả 4 nhóm (PII injection, jailbreak, off-topic,
prompt injection) với P95 172.6 ms, còn dư nhiều so với budget 500 ms.

**Về Cohen's κ = 0.4444 (moderate, chưa đạt ngưỡng substantial 0.6).** Nhãn judge lấy từ
`swap_and_average()` cho model_answer đấu với ground_truth, không phải từ ngưỡng token-overlap.
`verbosity_bias = 1.0` (`b_wins_b_longer: 7/7`) chỉ ra nguyên nhân: judge **luôn** chọn câu dài hơn,
mà ground_truth thì luôn dài hơn model answer. Hệ quả là judge không bao giờ cho model answer thắng —
3 nhãn `1` duy nhất đều đến từ `tie` do hai lượt swap bất đồng, chứ không phải judge thực sự đánh giá
model answer tốt hơn. Ba câu lệch (id 12, 23, 33) đều là câu người chấm cho đúng nhưng ngắn gọn.
Muốn κ lên thật thì phải sửa tiêu chí judge từ "câu nào tốt hơn" sang "câu này có đúng sự thật so với
reference không" — vì human label vốn đo tính đúng, không đo độ hay. Tinh chỉnh prompt cho tới khi
κ > 0.6 trên đúng 10 nhãn này là overfitting, không làm.

**Position bias 0.3** (3/10 cặp đảo kết quả khi swap) xác nhận swap-and-average là bắt buộc trong
production, không phải tuỳ chọn.

Trước production cần: nâng `RERANK_TOP_K` động theo loại truy vấn; thêm metadata `status`,
`effective_date`, `supersedes` để ưu tiên policy hiện hành (adversarial `context_recall` 0.6429 là
hệ quả trực tiếp của việc thiếu thứ này); timeout cứng + circuit breaker cho NeMo (P99 845 ms);
và giữ fail-closed khi guard service lỗi.
