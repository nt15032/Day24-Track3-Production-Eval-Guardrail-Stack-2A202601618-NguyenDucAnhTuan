# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Đức Anh Tuấn

**Ngày:** 26/08/2026

**Evaluator:** Offline deterministic proxy (token overlap); cần chạy lại hosted RAGAS khi có DeepSeek API key

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.7929 | 0.5791 | 0.8226 |
| context_precision | 0.9500 | 0.9333 | 0.7667 |
| context_recall | 0.8664 | 0.6584 | 0.5393 |
| **avg_score** | **0.9023** | **0.7927** | **0.7821** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | multi_hop | 40 | 0.5557 | context_recall |
| 2 | multi_hop | 37 | 0.5659 | answer_relevancy |
| 3 | adversarial | 50 | 0.6243 | context_recall |
| 4 | adversarial | 41 | 0.6637 | context_recall |
| 5 | adversarial | 48 | 0.6987 | context_precision |
| 6 | multi_hop | 39 | 0.7182 | answer_relevancy |
| 7 | multi_hop | 30 | 0.7202 | answer_relevancy |
| 8 | factual | 20 | 0.7320 | answer_relevancy |
| 9 | multi_hop | 33 | 0.7500 | context_recall |
| 10 | multi_hop | 34 | 0.7500 | answer_relevancy |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 5 | 0 | 0 | 5 |
| answer_relevancy | 8 | 13 | 1 | 22 |
| context_precision | 2 | 1 | 1 | 4 |
| context_recall | 5 | 6 | 8 | 19 |

## 4. Dominant Failure Analysis

**Dominant distribution:** adversarial

**Dominant metric:** answer_relevancy

Adversarial có điểm trung bình thấp nhất (0.7821), chủ yếu do context recall chỉ đạt 0.5393. Các câu về phiên bản chính sách và phủ định cần lấy đúng tài liệu hiện hành đồng thời loại tài liệu cũ. Xét toàn bộ 50 câu, answer relevancy là worst metric phổ biến nhất với 22 trường hợp; fallback trả nguyên chunk nên thường chứa đúng dữ kiện nhưng chưa tổng hợp thẳng vào câu hỏi multi-hop.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM có thể thêm thông tin ngoài context | Prompt bắt buộc trích nguồn, temperature 0 và output fact-check rail |
| context_recall | Thiếu chunk liên quan hoặc chỉ lấy một policy | Tăng candidate BM25, parent retrieval và truy vấn decomposition cho multi-hop |
| context_precision | Lấy cả policy cũ và mới | Lọc metadata theo trạng thái/ngày hiệu lực trước rerank |
| answer_relevancy | Fallback trả nguyên chunk thay vì tổng hợp | Dùng answer synthesis prompt theo từng ý và kiểm tra đủ sub-question |

## 6. Nhận xét về Adversarial Distribution

Adversarial (0.7821) thấp hơn factual (0.9023), đúng kỳ vọng stress-test. Ba câu adversarial xuất hiện trong bottom 10: ID 50 về VPN cá nhân, ID 41 về phép năm hiện hành và ID 48 về quyền lợi PVI của nhân viên thử việc. Nguyên nhân chính là version conflict và negation trap; metadata `effective_date/status` cùng policy precedence sẽ giúp retrieval ưu tiên bản hiện hành.

> Lưu ý: số liệu hiện tại được tạo bằng evaluator offline vì chưa có DeepSeek API key. Khi triển khai production, chạy lại `src/phase_a_ragas.py` bằng RAGAS/LLM judge trước khi dùng các ngưỡng làm CI gate.
