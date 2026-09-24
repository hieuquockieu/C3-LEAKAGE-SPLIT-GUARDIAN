# C3 – Leakage & Split Guardian

*Open Lab Challenge · Data Pipeline & Governance · Ngày 07*

---

## Slide 1 · Tiêu đề

# Leakage & Split Guardian

Phát hiện rò rỉ dữ liệu xuyên tập và xây dựng chính sách chia Train / Validation đáng tin cậy.

> **Validation score có đang "đẹp giả"?**

Nhóm [__] · Lớp [__] — Nhóm trưởng: Nguyễn Hoài Thanh · 7 thành viên

**Ghi chú thuyết trình:** Giới thiệu nhóm và đề tài C3 trong khoảng 10 giây, rồi đặt câu hỏi mở đầu và chuyển sang slide Pain point.

---

## Slide 2 · Thành viên nhóm

| # | Họ tên | Email | Vai trò |
|---|---|---|---|
| 1 | Nguyễn Hoài Thanh | 26ai.thanhnh@vinuni.edu.vn | Nhóm trưởng |
| 2 | Trần An Hạ | 26ai.hata@vinuni.edu.vn | Thành viên |
| 3 | Nguyễn Hữu Dũng | 26ai.dungnh@vinuni.edu.vn | Thành viên |
| 4 | Hoàng Tiến Dũng | 26ai.dunght@vinuni.edu.vn | Thành viên |
| 5 | Nguyễn Minh Tú | 26ai.tunm2@vinuni.edu.vn | Thành viên |
| 6 | Kiều Quốc Hiếu | 26ai.hieukq@vinuni.edu.vn | Thành viên |
| 7 | Ngô Xuân Nam | 26ai.namnx@vinuni.edu.vn | Thành viên |

---

## Slide 3 · Câu 1 – Pain point & cái giá của việc bỏ qua

### "Sát thủ vô hình" khi triển khai: rò rỉ dữ liệu xuyên tập

**Pain point cốt lõi**

- Dữ liệu thị giác thực tế (xe tự hành, giám sát giao thông, hệ thống nhiều camera) đến từ **chuỗi video liên tục, tuyến đường và cụm camera đa góc**.
- Random split giả định các mẫu độc lập (i.i.d.). Thực tế, frame liền kề và các góc camera cùng quay một chiếc xe bị rải vào **cả Train lẫn Validation**.

**Ai chịu hậu quả?**

- **ML Engineer:** "tự tin giả" vì validation "đẹp giả", tưởng model đã tổng quát hóa.
- **Product & vận hành:** ship model rồi tụt hiệu năng trên tuyến đường, camera chưa gặp.
- **Kinh doanh & an toàn:** tốn GPU train lại, trễ phát hành, rủi ro bỏ sót vật thể khi xe chạy thật.

| Con số | Ý nghĩa |
|---|---|
| **434** | cặp rò rỉ xuyên split trong random split |
| **1.0000** | cosine similarity cao nhất giữa Train và Val |
| **2 dạng** | temporal burst & multi-camera overlap |

**Ghi chú thuyết trình:** Validation có thể đang đo khả năng học thuộc, không phải khả năng tổng quát hóa. Đây là lỗi im lặng: không có error, training vẫn chạy, dashboard vẫn đẹp.

---

## Slide 4 · Câu 2 – Tái hiện lỗi & thiết kế benchmark

### Benchmark hai dạng rò rỉ, có ground truth manifest

- **Dạng 1 · Temporal burst overlap:** frame liên tiếp cách nhau ~150 ms, cùng một camera. Chỉ khác chút rung pixel, nên f0 ở Train và f1 ở Val khiến model học vẹt.
- **Dạng 2 · Multi-camera overlap:** nhiều camera cùng lúc ghi một sự kiện (cam_front_left / cam_front_right). Khác phối cảnh, ánh sáng nên pHash bỏ sót.

**Nhãn tham chiếu theo metadata, xét trên từng cặp Train–Val**

| Loại cặp | Quy tắc |
|---|---|
| Temporal (GT = 1) | Cùng scene, cùng camera, Δt nhỏ |
| Multi-view (GT = 1) | Cùng thời điểm, khác camera |
| Cặp âm (GT = 0) | Khác scene, đã kiểm tra không trùng bối cảnh |

**Holdout độc lập:** giữ riêng **scene 14–17** (**96 ảnh**) làm Test chưa từng thấy. Không dùng để train, chỉnh ngưỡng hay chọn model: đây là trọng tài kiểm tra điểm validation thật hay "đẹp giả".

**Ghi chú thuyết trình:** Nhãn tham chiếu là metadata proxy; cặp chưa đủ cơ sở để ở trạng thái "chưa xác định", không tự động tính là âm. Detector không dùng chính quy tắc metadata tạo nhãn rồi coi đó là bằng chứng nhận diện nội dung ảnh.

---

## Slide 5 · Câu 3 & 4 – Baseline thất bại & giải pháp SplitGuardian

### Kiến trúc: Hybrid Multimodal Graph Guardian

**Vì sao baseline thất bại**

- **Baseline 1 – Random split:** không biết gì về quan hệ thời gian hay không gian, tạo ra 434 cặp rò rỉ.
- **Baseline 2 – Perceptual hash:** bắt được ảnh trùng gần tuyệt đối nhưng bỏ sót góc camera khác. Recall 14,1%, F1 0,2128.

**Kiến trúc SplitGuardian**

1. **Trích xuất đặc trưng nhiều tầng:** dHash + embedding CNN 256 chiều, chuẩn hóa zero-mean.
2. **Đồ thị liên kết không–thời gian:** kết hợp độ giống hình ảnh với ràng buộc route, camera, timestamp.
3. **Tách connected component:** tìm các cụm rò rỉ nguyên tử.
4. **Stratified group re-splitter:** xếp nguyên cụm vào từng tập, các split tách rời 100% mà vẫn giữ cân bằng lớp.

> **F1 0,9393** · Recall 100% · so với baseline pHash F1 0,2128 → **cải thiện +341%**

**Ghi chú thuyết trình:** Chuẩn bị Q&A: đồ thị có dùng metadata route/camera/timestamp, nên cần nói rõ phần nào của F1 đến từ nội dung ảnh, phần nào từ metadata.

---

## Slide 6 · Câu 5 – Bằng chứng định lượng

### Điểm trung thực và điểm bị thổi phồng

Đơn vị: cặp ảnh Train–Val · P = TP/(TP+FP) · R = TP/(TP+FN) · F1 = 2PR/(P+R)

| Phương pháp phát hiện | Precision | Recall | F1 | Rò rỉ còn lại | Tốc độ |
|---|---|---|---|---|---|
| Perceptual hash (baseline) | 0,4323 | 0,1412 | 0,2128 | 372 cặp | 388 ảnh/s |
| Deep visual embedding | 0,4246 | 0,4881 | 0,4541 | 221 cặp | 365 ảnh/s |
| **SplitGuardian (hybrid graph)** | **0,8855** | **1,0000** | **0,9393** | **0 cặp (sạch)** | **250 ảnh/s** |

| | Validation | Holdout test | Nhận xét |
|---|---|---|---|
| **Leaky split** | 100,0% (quá lạc quan) | 94,8% | Khoảng cách tổng quát hóa: +5,2 điểm tự tin giả |
| **Clean split** | 91,2% (phản ánh thật) | 97,9% | Loại bỏ "đẹp giả" mà không làm giảm chất lượng model |

**Ghi chú thuyết trình:** Điểm validation thấp hơn sau khi chia sạch không có nghĩa model tệ hơn; đánh giá trên cùng một holdout độc lập cho thấy clean split đáng tin hơn. Holdout chỉ 96 ảnh, nên nói rõ số seed đã chạy.

---

## Slide 7 · Câu 6 – Quyết định production, cổng kiểm tra & đánh đổi

### Quyết định: DEPLOY kèm cổng kiểm tra trong CI/CD

- **Quyết định: DEPLOY** – `split_guardian_gate.py` trong CI/CD: fail build (exit code 1) khi còn cặp rò rỉ hoặc similarity cao nhất vượt ngưỡng.
- **Đánh đổi khi chia lại** – cụm burst lớn có thể làm lệch phân bố lớp; bin-packing phân tầng giữ độ lệch dưới 4%. Tập Val nhỏ và dao động hơn.
- **Giới hạn & tiếp theo** – người kiểm duyệt cặp similarity vùng biên 0,92–0,96; mở rộng kiểm tra streaming cho đội xe tự hành.

**Quy trình kiểm tra trước huấn luyện:** Manifest dữ liệu → Kiểm tra chia tập → Xem xét cảnh báo → Huấn luyện

> **Độ tin cậy của kết quả phụ thuộc vào ranh giới dữ liệu và cách đánh giá, không chỉ con số validation.**

**Ghi chú thuyết trình:** Giao thoa scene là lỗi chặn; similarity cao là cảnh báo cần người xem xét, không tự động xóa ảnh. Lưu phiên bản dataset, danh sách scene, seed, cấu hình bộ dò để tái lập. Có thể mở web dashboard trong phần Q&A (không demo thay cho số liệu).
