# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Đức Anh Tuấn

**Ngày:** 26/08/2026

**Judge:** Offline deterministic fallback; cấu hình production là `gpt-4o-mini`

## 1. Pairwise Judge Results

| # | Question ID | Winner | Reasoning tóm tắt |
|---:|---:|---|---|
| 1 | 1 | B | B bổ sung việc không trừ phép năm |
| 2 | 5 | B | B nêu đúng ngưỡng trên 50 triệu và CEO |
| 3 | 12 | B | B đầy đủ điều kiện 6 tháng và pro-rata |
| 4 | 21 | A | A súc tích nhưng vẫn đủ phép và khung lương |
| 5 | 23 | B | B nêu rõ cam kết, tỷ lệ hoàn trả và số tiền |

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | B | B | B | Yes |
| 2 | B | B | B | Yes |
| 3 | B | B | B | Yes |
| 4 | A | A | A | Yes |
| 5 | B | B | B | Yes |

**Position bias rate:** 0% (0/5).

## 3. Cohen's κ Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 1 | No |
| 12 | 1 | 0 | No |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.4000

**Interpretation:** moderate agreement; chưa đạt bonus κ > 0.6.

## 4. Verbosity Bias

- Answer thắng và cũng dài hơn: 4/5 decisive cases.
- **Verbosity bias rate:** 80%.

Tỷ lệ cao cho thấy judge fallback đang thưởng quá nhiều cho độ đầy đủ bề mặt. Một câu dài có thể chứa nhiều từ khóa nhưng vẫn sai policy. Production judge cần rubric riêng cho factual correctness, giới hạn điểm completeness và luôn swap vị trí.

## 5. Nhận xét chung

Swap-and-average loại được position bias trong mẫu này, nhưng κ=0.4 cho thấy judge fallback chưa đủ tin cậy để tự động chặn merge. Các lỗi tập trung ở câu trả lời ngắn nhưng đúng và câu trả lời có nhiều từ trùng nhưng sai authority. Khi có API key, cần chạy lại bằng `gpt-4o-mini` với structured JSON, temperature 0 và human calibration set; chỉ dùng judge như một CI signal kết hợp RAGAS và review con người.
