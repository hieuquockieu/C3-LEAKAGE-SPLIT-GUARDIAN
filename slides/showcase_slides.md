# Chủ Đề C3: Leakage & Split Guardian — Báo Cáo Thuyết Trình 5 Slide

## Slide 1: Bối Cảnh Bài Toán, Pain Point Cụ Thể & Hậu Quả (Question 1)
- **Pain point cụ thể:**
  - Trong các bài toán thị giác máy tính thực tế (xe tự hành, camera giao thông, giám sát an ninh), dữ liệu luôn thu thập theo chuỗi thời gian (video sequences), chụp liên tiếp (frame bursts), hoặc hệ thống nhiều góc nhìn (multi-camera rig).
  - Khi áp dụng phân chia ngẫu nhiên (Naive Random Split), các khung hình liền kề ($t \pm 0.1s$) hoặc góc nhìn đối diện của cùng một phương tiện/sự kiện bị xáo trộn vào cả tập Train và Validation.
- **Ai chịu hậu quả nếu không giải quyết?**
  - **ML Engineers:** Bị ảo tưởng bởi validation metric "đẹp giả" (96-100%), không phát hiện được hiện tượng overfit vào các khung hình trùng lặp.
  - **Product & Vận hành:** Khi đưa mô hình ra thực tế (unseen routes / unseen cameras), hiệu năng tụt dốc thảm hại (giảm 20-40%), gây gián đoạn dịch vụ.
  - **Doanh nghiệp & An toàn:** Lãng phí hàng trăm giờ GPU huấn luyện lại, nguy cơ tai nạn nghiêm trọng trong các hệ thống an toàn sinh mạng (ADAS / Autonomous Vehicles).

---

## Slide 2: Failure Reproduction & Benchmark Thiết Kế (Question 2)
- **Thiết kế Benchmark Dataset (432 samples, 18 scenes, 4 classes):**
  - Có Ground Truth manifest rõ ràng (`metadata.json`, `ground_truth_leakage_pairs.json`).
  - Gồm 14 Train/Val Scenes (336 samples) và 4 Independent Holdout Test Scenes (96 samples).
- **Ít nhất 2 dạng Leakage được chứng minh thực nghiệm:**
  1. **Leakage Type 1 - Temporal Burst Overlap:** Các khung hình liên tiếp cách nhau 150ms từ cùng camera. Sai khác chỉ ở độ rung pixel và góc dịch chuyển nhỏ.
  2. **Leakage Type 2 - Multi-Camera Viewpoint Overlap:** Cùng một sự kiện phương tiện được ghi hình đồng thời từ `cam_front_left` và `cam_front_right`.
- **Hậu quả của Naive Random Split:** Tạo ra tới **434 cặp rò rỉ xuyên tập (Cross-Split Leaked Pairs)** với Cosine Similarity đạt tới $1.0000$!

---

## Slide 3: Baseline vs Solution SplitGuardian (Questions 3 & 4)
- **Baseline là gì và tệ đến đâu?**
  - *Baseline 1 (Random Split):* Mù hoàn toàn trước cấu trúc dữ liệu, tạo ra 434 cặp rò rỉ.
  - *Baseline 2 (Perceptual Hash - pHash/dHash):* Chỉ nhận diện được bản sao thô sơ, bỏ sót hầu hết các góc nhìn đa camera (Recall chỉ 14.12%, F1 chỉ 0.2128).
- **Giải pháp SplitGuardian:**
  1. *Multi-tier Feature Extraction:* Kết hợp dHash và 256-D Deep CNN Feature Vector đã được chuẩn hóa zero-mean.
  2. *Spatio-Temporal Affinity Graph:* Tích hợp ràng buộc định danh chuỗi, camera ID và cửa sổ thời gian.
  3. *Connected Component Clustering:* Gom các mẫu phụ thuộc thành các "Atomic Leakage Clusters".
  4. *Guardian Group-Preserving Stratified Split:* Phân bổ nguyên cụm vào Train hoặc Val theo giải thuật Greedy Bin-Packing giữ vững cân bằng phân phối lớp.

---

## Slide 4: Bằng Chứng Thực Nghiệm Định Lượng (Question 5)
- **Bảng so sánh Primary & Secondary Metrics:**

| Phương pháp | Precision | Recall | F1 Score | Rò rỉ còn lại | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Perceptual Hash (Baseline) | 0.4323 | 0.1412 | 0.2128 | 372 cặp | 388 img/s |
| Deep Feature Embeddings | 0.4246 | 0.4881 | 0.4541 | 221 cặp | 365 img/s |
| **SplitGuardian (Hybrid Graph)** | **0.8855** | **1.0000** | **0.9393** | **0 cặp (Clean)** | **250 img/s** |

- **Chứng minh "Đẹp Giả" (Generalization Gap Proof):**
  - **Leaky Split:** Validation Acc = **100.0%**, Holdout Test Acc = **94.8%** $\rightarrow$ Độ lệch lạc quan giả = **+5.2%**.
  - **Clean Split:** Validation Acc = **91.2%**, Holdout Test Acc = **97.9%** $\rightarrow$ Metric phản ánh trung thực năng lực khái quát hóa trên dữ liệu thực tế!

---

## Slide 5: Quyết Định Triển Khai Sản Phẩm & Lộ Trình (Question 6)
- **Production Decision: DEPLOY**
  - Tích hợp cổng kiểm soát tự động `src.cli.split_guardian_gate` vào CI/CD pipeline (trước commit/trước training).
  - Tự động chặn build (Exit Code 1) khi phát hiện rò rỉ dữ liệu hoặc vượt ngưỡng tương đồng cho phép.
- **Trade-off khi Re-split:**
  - *Trade-off phân phối lớp:* Khi dữ liệu có ít cụm lớn, việc chia nguyên cụm có thể làm lệch nhẹ phân phối lớp. SplitGuardian giải quyết bằng cơ chế phân chia phân tầng theo lớp chiếm ưu thế, duy trì độ lệch dưới 4%.
- **Giới hạn & Bước tiếp theo:**
  - Thiết lập Human-in-the-loop review đối với các cặp có độ tương đồng mấp mé ngưỡng $0.92-0.96$.
  - Mở rộng kiến trúc xử lý dạng streaming cho các luồng ingestion dữ liệu lớn theo thời gian thực.
