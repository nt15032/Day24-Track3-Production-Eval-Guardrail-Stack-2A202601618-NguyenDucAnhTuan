# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Đức Anh Tuấn

**Ngày:** 27/08/2026

**Judge:** `gpt-4o-mini`, `temperature=0`, structured output JSON schema (`strict: true`).
`reports/judge_results.json` → `judge_model: gpt-4o-mini`. Không dùng fallback offline.

**Thiết lập:** mỗi câu trong `human_labels_10q.json` được chấm bằng `swap_and_average()` với
**A = model_answer**, **B = ground_truth**. Nhãn judge = 1 nếu model answer thắng hoặc hoà
reference, = 0 nếu thua reference. Toàn bộ 10 câu đều đi qua 2 pass ⇒ 20 lần gọi LLM.

## 1. Swap-and-Average Results (10/10 câu)

| # | Question ID | Pass 1 | Pass 2 (đã quy đổi) | Final | Position Consistent? |
|---:|---:|---|---|---|---|
| 1 | 1 | B | tie | tie | No |
| 2 | 5 | B | B | B | Yes |
| 3 | 12 | B | B | B | Yes |
| 4 | 21 | B | A | tie | No |
| 5 | 23 | B | B | B | Yes |
| 6 | 29 | B | B | B | Yes |
| 7 | 33 | B | B | B | Yes |
| 8 | 41 | B | B | B | Yes |
| 9 | 46 | B | A | tie | No |
| 10 | 50 | B | B | B | Yes |

**Position bias rate: 0.30 (3/10).**

Phát hiện quan trọng nhất nằm ở cột Pass 1: **B thắng 10/10**. Judge chưa một lần nào chọn model
answer khi nó đứng ở vị trí A. Ba kết quả `tie` duy nhất đều sinh ra do hai pass bất đồng rồi bị
`swap_and_average()` ép hoà — **không phải** vì judge thực sự đánh giá model answer ngang reference.

## 2. Verbosity Bias

| Chỉ số | Giá trị |
|---|---:|
| Decisive cases (loại `tie`) | 7 |
| Câu thắng đồng thời dài hơn | 7 |
| `a_wins_a_longer` | 0 |
| `b_wins_b_longer` | 7 |
| **Verbosity bias** | **1.00** |

Tuyệt đối 7/7. Trong khi prompt của `pairwise_judge()` đã ghi rõ *"So sánh theo độ chính xác, đầy đủ
và súc tích. Không ưu tiên câu dài hơn."*, lý do judge đưa ra lại nói ngược lại — trích nguyên văn:

- id 1 — *"Answer B provides **additional information** that the leave does not deduct from annual leave"*
- id 12 — *"Câu trả lời B cung cấp thông tin **đầy đủ hơn** bằng cách nêu rõ điều kiện về thời gian làm việc"*
- id 33 — *"Answer B provides a **detailed breakdown** of the monthly allowance"*
- id 41 — *"Answer B provides a **more comprehensive** response"*

Cả 10 reasoning đều viện dẫn tính đầy đủ/chi tiết. Vì B luôn là `ground_truth` và ground_truth luôn
dài hơn model answer, tiêu chí "đầy đủ" biến thành tiêu chí "dài hơn" — chỉ dẫn phủ định trong prompt
không đủ để chặn.

## 3. Cohen's κ Analysis

| Question ID | Human Label | Judge Label | Judge Winner | Agree? |
|---:|---:|---:|---|---|
| 1 | 1 | 1 | tie | Yes |
| 5 | 0 | 0 | B | Yes |
| 12 | 1 | 0 | B | **No** |
| 21 | 1 | 1 | tie | Yes |
| 23 | 1 | 0 | B | **No** |
| 29 | 0 | 0 | B | Yes |
| 33 | 1 | 0 | B | **No** |
| 41 | 0 | 0 | B | Yes |
| 46 | 1 | 1 | tie | Yes |
| 50 | 0 | 0 | B | Yes |

Observed agreement `p_o` = 7/10 = 0.70
Human: 6×`1`, 4×`0` · Judge: 3×`1`, 7×`0`
`p_e` = 0.6×0.3 + 0.4×0.7 = 0.46
**κ = (0.70 − 0.46) / (1 − 0.46) = 0.4444**

**Interpretation:** moderate agreement (Landis–Koch 0.4–0.6). Chưa đạt bonus κ > 0.6.

Cả 3 câu lệch (id 12, 23, 33) đều cùng một dạng: **người chấm cho đúng, judge cho sai**, và cả 3 đều
là câu trả lời ngắn gọn nhưng chính xác. Không có câu nào lệch theo chiều ngược lại. Sai số của judge
có hướng rõ ràng, không phải nhiễu ngẫu nhiên — đúng như verbosity bias 1.00 dự đoán.

## 4. Vì sao không "chỉnh" prompt cho κ vượt 0.6

Chỉ có 10 nhãn người. Sửa prompt rồi đo lại trên đúng 10 nhãn đó, lặp cho tới khi κ > 0.6, là
**overfitting vào tập đánh giá** — con số đẹp nhưng không nói lên judge tốt hơn thật.

Hướng sửa đúng là đổi *tiêu chí*, không phải dò tham số: human label đo **tính đúng sự thật**
("Sai — 55 triệu vượt ngưỡng 50 triệu nên phải cấp cao hơn duyệt"), trong khi judge hiện đo **câu nào
hay hơn**. Hai thang đo khác nhau thì κ thấp là tất yếu. Muốn đo cùng một thứ, `pairwise_judge()` phải
đổi thành câu hỏi kiểu factual-entailment — *"câu trả lời này có mâu thuẫn với reference không?"* —
thay vì *"câu nào tốt hơn?"*. Việc này thay đổi hợp đồng của Task 5 nên để lại như hướng cải tiến.

## 5. Nhận xét chung

1. **Swap-and-average là bắt buộc, không phải tuỳ chọn.** 3/10 cặp đảo kết quả khi hoán vị. Nếu chỉ
   chạy một pass, 3 câu đó sẽ bị gán nhãn sai một cách âm thầm.
2. **Verbosity bias 1.00 là rủi ro production thật.** Judge kiểu này sẽ thưởng cho câu trả lời dài
   dòng, đẩy hệ thống về phía sinh câu dài để "ăn điểm" thay vì trả lời đúng và gọn.
3. **κ = 0.4444 chưa đủ để tự động chặn merge.** Ở mức moderate, judge chỉ nên dùng làm tín hiệu CI
   phụ, kết hợp RAGAS và review người. Ngưỡng an toàn để tự động hoá là κ > 0.6 trên calibration set
   lớn hơn 10 mẫu.
4. **Cần mở rộng calibration set.** Với n = 10, một nhãn đổi làm κ dịch khoảng 0.1 — sai số quá lớn
   để kết luận chắc chắn về chất lượng judge.
