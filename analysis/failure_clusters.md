# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Đức Anh Tuấn

**Ngày:** 27/08/2026

**Evaluator:** RAGAS 0.1.22 hosted — LLM judge `deepseek-chat`, embeddings `BAAI/bge-m3` (GPU A100).
`reports/ragas_50q.json` → `evaluation_mode: ragas_deepseek`. Không dùng proxy offline.

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial | Toàn bộ 50q |
|---|---:|---:|---:|---:|
| faithfulness | 0.9629 | 0.7326 | 0.8179 | 0.8418 |
| answer_relevancy | 0.8879 | 0.6615 | 0.5148 | 0.7227 |
| context_precision | 1.0000 | 0.9833 | 0.9333 | 0.9800 |
| context_recall | 0.9544 | 0.7545 | 0.6429 | 0.8121 |
| **avg_score** | **0.9513** | **0.7830** | **0.7272** | **0.8392** |

Thứ tự suy giảm `factual → multi_hop → adversarial` đúng như thiết kế stress-test của đề.

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | adversarial | 48 | 0.4220 | answer_relevancy |
| 2 | multi_hop | 33 | 0.4524 | answer_relevancy |
| 3 | multi_hop | 21 | 0.4649 | answer_relevancy |
| 4 | adversarial | 50 | 0.4921 | answer_relevancy |
| 5 | multi_hop | 32 | 0.5039 | faithfulness |
| 6 | adversarial | 42 | 0.5531 | answer_relevancy |
| 7 | adversarial | 44 | 0.7059 | answer_relevancy |
| 8 | adversarial | 45 | 0.7083 | answer_relevancy |
| 9 | multi_hop | 40 | 0.7530 | context_precision |
| 10 | multi_hop | 34 | 0.7711 | answer_relevancy |

Không có câu `factual` nào lọt bottom 10 — toàn bộ 10 câu tệ nhất là `multi_hop` (4) và
`adversarial` (6), dù `adversarial` chỉ chiếm 10/50 câu của test set.

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 10 | 7 | 0 | 17 |
| answer_relevancy | 8 | 7 | 6 | 21 |
| context_precision | 0 | 1 | 0 | 1 |
| context_recall | 2 | 5 | 4 | 11 |

## 4. Dominant Failure Analysis

**Dominant distribution:** adversarial (avg 0.7272 — thấp nhất)

**Dominant metric:** answer_relevancy (21/50 câu, worst metric phổ biến nhất)

`answer_relevancy` sụt theo đúng độ khó: 0.8879 (factual) → 0.6615 (multi_hop) → **0.5148**
(adversarial). Đây là chỉ số duy nhất rơi xuống dưới 0.55 ở bất kỳ nhóm nào.

Điểm đáng chú ý nhất là tương phản giữa hai metric context:

- `context_precision` **0.9800** — gần như hoàn hảo ở cả 3 nhóm
- `context_recall` **0.8121**, riêng adversarial chỉ 0.6429

Reranker lọc rất sạch nhưng **lọc quá tay**: những gì lấy về đều đúng, chỉ là chưa lấy đủ. Với
`RERANK_TOP_K = 3` cố định, câu multi-hop cần 4–5 đoạn (ví dụ tính lương thử việc phải ghép
`thu_viec.md` + `bang_luong_2024.md` + `phu_cap.md`) sẽ bị cắt mất phần cần thiết trước khi LLM
kịp tổng hợp. Recall thiếu → câu trả lời không phủ hết ý hỏi → `answer_relevancy` tụt theo.

Riêng `faithfulness` ở nhóm `factual` bị tính là worst metric 10 lần nhưng giá trị tuyệt đối vẫn
cao (0.9629) — đó là hệ quả của việc lấy `min()` trên 4 metric đều cao, không phải dấu hiệu
hallucinate. Đọc matrix phải kèm giá trị tuyệt đối, không chỉ đếm số lần.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| answer_relevancy | Context thiếu ý → câu trả lời không phủ hết câu hỏi; prompt không ép trả lời từng vế | Query decomposition cho multi-hop, prompt liệt kê rõ từng sub-question và bắt trả lời đủ |
| context_recall | `RERANK_TOP_K = 3` cứng, cắt mất đoạn cần cho câu ghép nhiều tài liệu | Top-k động theo loại câu hỏi (3 cho factual, 5–7 cho multi_hop); bật parent retrieval để trả về parent chunk thay vì child |
| context_precision | Đã tốt (0.98), không cần can thiệp | Giữ nguyên cross-encoder `bge-reranker-v2-m3` |
| faithfulness (multi_hop 0.7326) | Khi context thiếu, LLM tự suy luận để lấp chỗ trống | Prompt bắt buộc trích nguồn theo từng câu; trả "Không tìm thấy" thay vì suy đoán |

## 6. Nhận xét về Adversarial Distribution

Adversarial (**0.7272**) thấp hơn factual (**0.9513**) — chênh 0.2241, đúng kỳ vọng stress-test và
đạt điều kiện bonus Phase A.

6/10 câu bottom-10 là adversarial: ID 48 (quyền lợi PVI của nhân viên thử việc), ID 50 (VPN cá
nhân), ID 42, 44, 45. Toàn bộ đều thuộc hai kiểu bẫy của đề:

- **Version conflict** — corpus cố ý chứa `nghi_phep_nam_v2023.md` cạnh `nghi_phep_nam_v2024.md`,
  `mat_khau_v1.md` cạnh `mat_khau_v2.md`. Retrieval hiện không phân biệt bản nào còn hiệu lực nên
  kéo về cả hai, `context_recall` adversarial vì thế chỉ 0.6429.
- **Negation trap** — câu hỏi dạng "có nên tự xử lý không?" cần trả lời phủ định dứt khoát, nhưng
  chunk gốc mô tả quy trình theo hướng khẳng định.

Cách sửa gốc là thêm metadata `status`, `effective_date`, `supersedes` cho từng tài liệu rồi lọc
theo trạng thái **trước** khi rerank, thay vì kỳ vọng cross-encoder tự đoán ra bản nào mới hơn.
