# SplitGuardian: Leakage & Split Policy Guardian (Topic C3)

> **Challenge C3 — Production Question:**  
> *"Detect cross-split leakage, định lượng severity và tạo/đề xuất split policy đáng tin hơn.  
> Liệu Validation score có 'đẹp giả' do overlap/leakage chưa?"*

---

## 📋 Mục Lục
1. [Sáu Câu Hỏi Cốt Lõi (Mandatory 6 Questions)](#1-sáu-câu-hỏi-cốt-lõi-mandatory-6-questions)
   - [Q1: Pain point cụ thể & Ai chịu hậu quả?](#q1-pain-point-cụ-thể--ai-chịu-hậu-quả)
   - [Q2: Phương pháp reproduce & Chứng minh failure](#q2-phương-pháp-reproduce--chứng-minh-failure)
   - [Q3: Baseline là gì & Hoạt động tệ đến đâu?](#q3-baseline-là-gì--hoạt-động-tệ-đến-đâu)
   - [Q4: Giải pháp SplitGuardian & Kiến trúc hệ thống](#q4-giải-pháp-splitguardian--kiến-trúc-hệ-thống)
   - [Q5: Bằng chứng định lượng (Evidence Before vs After)](#q5-bằng-chứng-định-lượng-evidence-before-vs-after)
   - [Q6: Quyết định sản phẩm (Production Decision, Limits, Roadmap)](#q6-quyết-định-sản-phẩm-production-decision-limits-roadmap)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [Hướng Dẫn Tái Lập Thực Nghiệm (Step-by-step Reproduction)](#3-hướng-dẫn-tái-lập-thực-nghiệm-step-by-step-reproduction)
4. [Dataset, Protocol & License](#4-dataset-protocol--license)
5. [Ablation Study, Phân Tích Chi Phí & Runtime](#5-ablation-study-phân-tích-chi-phí--runtime)
6. [Interactive Web Dashboard & 5-Slide Showcase](#6-interactive-web-dashboard--5-slide-showcase)

---

## 1. Sáu Câu Hỏi Cốt Lõi (Mandatory 6 Questions)

### Q1: Pain point cụ thể & Ai chịu hậu quả?
- **Pain point kỹ thuật:**  
  Trong các bài toán Computer Vision thực tế (xe tự hành, camera giám sát giao thông, robot giao hàng), dữ liệu luôn được ghi nhận dưới dạng chuỗi video liên tục (video sequences), cụm chụp liên tiếp (frame bursts), hoặc hệ thống camera đa góc nhìn (multi-camera rig).  
  Phương pháp phân chia dữ liệu truyền thống (Naive Random Split) giả định ngây thơ rằng các mẫu là *độc lập và phân phối đồng nhất (i.i.d.)*. Do đó, các khung hình liền kề chỉ cách nhau vài mili-giây ($t \pm 0.1s$) hoặc góc nhìn đồng thời từ camera trái và phải của cùng một vật thể bị chia rẽ ngẫu nhiên vào cả tập **Train** và tập **Validation**.
- **Ai chịu hậu quả nếu không giải quyết?**
  - **ML Engineers:** Bị ảo tưởng bởi metric validation "đẹp giả" (Validation Accuracy / mAP đạt 96–100%), tin rằng mô hình đã hội tụ và tổng quát hóa tốt.
  - **Đội ngũ Vận hành & Sản phẩm:** Khi triển khai mô hình ra môi trường thực tế (tuyến đường mới, góc camera mới), hiệu năng mô hình sụp đổ bất ngờ (tụt giảm 20–40% mAP), gây đình trệ sản phẩm.
  - **Doanh nghiệp & An toàn:** Lãng phí hàng ngàn giờ tính toán GPU để train lại từ đầu; nguy cơ tai nạn nghiêm trọng trong các hệ thống an toàn tối quan trọng (ADAS / Xe tự hành không nhận diện được xe lạ trên đường).

---

### Q2: Phương pháp reproduce & Chứng minh failure?
Chúng tôi xây dựng bộ dữ liệu benchmark giả lập chuẩn công nghiệp (`dataset_store`) với metadata chi tiết và Ground Truth nhãn rò rỉ:
- **Quy mô:** 432 captures trải dài trên 18 scenes độc lập, 4 lớp đối tượng (`sedan`, `truck`, `motorcycle`, `emergency_vehicle`), 2 camera đồng bộ (`cam_front_left`, `cam_front_right`), và chuỗi burst 4 frame (150ms/frame).
- **Chứng minh 2 Dạng Leakage cụ thể:**
  1. **Leakage Dạng 1 — Temporal Burst Overlap:** Các khung hình trong cùng chuỗi thời gian của cùng 1 camera (sai khác chủ yếu do nhiễu pixel và vi dịch chuyển $150\text{ ms}$).
  2. **Leakage Dạng 2 — Multi-Camera Viewpoint Overlap:** Cùng một sự kiện phương tiện được ghi hình đồng thời từ 2 góc nhìn camera khác nhau với độ lệch góc phối cảnh và ánh sáng.
- **Minh chứng Failure của Naive Random Split:**
  - Tạo ra tới **434 cặp rò rỉ xuyên tập (Cross-Split Leaked Pairs)** giữa Train và Val.
  - Độ tương đồng Cosine cực đại giữa tập Train và Val đạt **1.0000** (hai mẫu thực chất là bản sao của nhau).
  - Validation Accuracy đạt **100.0%** ("đẹp giả"), nhưng khi test trên Independent Holdout Test Set (Scenes 14–17 hoàn toàn mới lạ), mô hình bộc lộ Generalization Gap lên đến **+5.2%**.

---

### Q3: Baseline là gì & Hoạt động tệ đến đâu?
1. **Baseline 1 — Naive Random Split (Mù cấu trúc):**
   - Xáo trộn ngẫu nhiên từng ảnh độc lập.
   - *Kết quả:* 434 cặp rò rỉ xuyên tập, không có cơ chế phát hiện hay ngăn chặn rò rỉ.
2. **Baseline 2 — Perceptual Hash (aHash / dHash / pHash với ngưỡng Hamming):**
   - Trích xuất mã băm gradient (dHash 256-bit), so sánh khoảng cách Hamming.
   - *Điểm yếu:* Chỉ phát hiện được các bản sao thô sơ trong cùng camera. Khi gặp **Multi-Camera Viewpoint Shift** (thay đổi góc nhìn phối cảnh), khoảng cách Hamming tăng vọt vượt ngưỡng, dẫn đến việc bỏ sót nghiêm trọng.
   - *Hiệu năng Primary Metric:* **Precision: 0.4323 | Recall: 0.1412 | F1: 0.2128** (bỏ sót hơn 85% các cặp rò rỉ góc nhìn đa camera).

---

### Q4: Giải pháp SplitGuardian & Kiến trúc hệ thống
Hệ thống **SplitGuardian** giải quyết bài toán qua 4 tầng xử lý:

```
[ Input Dataset + Metadata ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multi-Tier Feature Extraction                       │
│    - Fast Perceptual Hash (dHash 256-bit)              │
│    - Deep Zero-Mean Calibrated CNN Embeddings (256-D)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Spatio-Temporal Affinity Graph Engine               │
│    - Route/Scene spatial boundary isolation            │
│    - Temporal proximity window (Δt ≤ 15s)              │
│    - Concurrent multi-camera observation co-weight     │
│    - Fused Affinity: W = 0.50*Emb + 0.25*Hash + 0.25*Meta│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Connected Component Graph Clustering                │
│    - Discover Atomic Leakage Units (Clusters)          │
│    - Zero leak between distinct scenes                 │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Group-Preserving Stratified Re-Split Engine         │
│    - Atomic cluster isolation (All-in-Train or All-in-Val)│
│    - Greedy Class-Stratified Bin-Packing               │
│    - Target ratio: 75% Train, 25% Val                  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
 [ Verified Clean Split (0 Leaks) + CI/CD Gate Report ]
```

- **Lý do chọn kỹ thuật:**
  - *Zero-Mean Calibrated Embeddings:* Khắc phục triệt để nhược điểm cosine similarity bị méo ở các vector không âm.
  - *Affinity Graph Fusion:* Bổ sung ràng buộc metadata giúp hệ thống đạt độ bao phủ (Recall) tuyệt đối mà không bị false positive giữa các xe cùng màu ở thành phố khác nhau.
  - *Group-Preserving Stratified Split:* Bảo đảm 100% cách ly các cụm rò rỉ mà vẫn duy trì phân phối nhãn đồng đều giữa Train và Val.

---

### Q5: Bằng chứng định lượng (Evidence Before vs After)

#### Bảng 1: So sánh Primary Metric (Leakage Detection Performance)
| Phương pháp | Precision | Recall | Primary F1 | Cross Leaks Còn Lại | Runtime (s) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Random Split** | 0.0000 | 0.0000 | 0.0000 | 434 cặp | 0.01s | N/A |
| **Baseline 2: Perceptual Hash** | 0.4323 | 0.1412 | 0.2128 | 372 cặp | 0.64s | 388 img/s |
| **Deep Visual Embeddings** | 0.4246 | 0.4881 | 0.4541 | 221 cặp | 3.08s | 365 img/s |
| **⭐ SplitGuardian (Hybrid Graph)** | **0.8855** | **1.0000** | **0.9393** | **0 cặp (Clean)** | **4.52s** | **250 img/s** |

> **Primary Metric Target:** SplitGuardian đạt **F1 = 0.9393** (+341% so với baseline), đạt **100% Recall** trên cả 2 dạng rò rỉ (Temporal Burst và Multi-Camera).

#### Bảng 2: So sánh Secondary Metric (Cross-Split Similarity & Model Reliability)
| Chiến lược Split | Cross Leaked Pairs | Max Cosine Sim | Mean Cosine Sim | Val Acc (%) | Holdout Test Acc (%) | Gen Gap (%) | Đánh giá độ tin cậy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Random Split** | 434 cặp | 1.0000 | -0.0005 | 100.0% | 94.8% | **+5.2%** | 🔴 **ĐẸP GIẢ (Overfit)** |
| **SplitGuardian Re-Split** | **0 cặp** | **0.9550** | **-0.0259** | 91.2% | **97.9%** | **-6.7%** | 🟢 **ĐÁNG TIN (Honest)** |

> **BẪY / LƯU Ý ĐƯỢC GIẢI QUYẾT:**  
> Validation score trên Clean Split (91.2%) thấp hơn so với Leaky Split (100.0%) không có nghĩa là model tệ hơn! Trên tập **Holdout Test Set hoàn toàn độc lập (unseen scenes)**, model được huấn luyện trên Clean Split đạt **97.9%** (cao hơn model leaky đạt 94.8%).  
> Điều này chứng minh: Phân chia sạch buộc mô hình phải học đặc trưng hình học bất biến thật sự thay vì học vẹt khung hình rò rỉ!

---

### Q6: Quyết định sản phẩm (Production Decision, Limits, Roadmap)
- **Quyết định: DEPLOY**
  - Đưa công cụ `src/cli/split_guardian_gate.py` vào CI/CD pipeline của đội ngũ MLOps.
  - Tự động kích hoạt kiểm tra trước mỗi commit dữ liệu hoặc trước khi cấp tài nguyên GPU huấn luyện. Nếu phát hiện rò rỉ vượt ngưỡng (`max_allowed_leaked_pairs > 0`), pipeline lập tức **BLOCK BUILD** và xuất báo cáo `reports/ci_gate_report.json`.
- **Mô tả Trade-off khi Re-split:**
  - *Trade-off phân phối lớp:* Khi dữ liệu có ít cụm lớn, việc phân bổ nguyên cụm vào một tập có thể gây chênh lệch tỷ lệ nhãn lớp. SplitGuardian giải quyết bằng giải thuật phân tầng theo cụm (Stratified Cluster Bin-Packing), giữ mức chênh lệch tỷ lệ lớp dưới 4%.
  - *Trade-off kích thước tập:* Kích thước tập Val thực tế có thể dao động nhẹ ($\pm 2\%$) xung quanh target ratio để bảo đảm tính nguyên khối của cụm.
- **Giới hạn & Bước tiếp theo:**
  - *Giới hạn hiện tại:* Cần cơ chế Human-in-the-loop audit cho các cặp có độ tương đồng mấp mé ngưỡng cảnh báo ($0.92 - 0.96$).
  - *Roadmap tiếp theo:* Tích hợp streaming check bằng Vector Index (FAISS/HNSW) cho các luồng dữ liệu video streaming quy mô hàng triệu khung hình.

---

## 2. Cấu Trúc Thư Mục Dự Án

```
leakage-split-guardian/
├── README.md                           # Toàn bộ tài liệu báo cáo & hướng dẫn tái lập
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── configs/
│   └── guardian_policy.yaml            # Cấu hình chính sách sản xuất (ngưỡng, tỷ lệ, gate)
├── data/
│   └── dataset_generator.py            # Sinh benchmark đa camera/video burst & ground truth
├── dataset_store/
│   ├── metadata.json                   # Manifest mẫu dữ liệu và nhãn lớp
│   ├── ground_truth_leakage_pairs.json # Danh sách đầy đủ các cặp rò rỉ ground truth
│   └── images/                         # Thư mục ảnh đã sinh
├── src/
│   ├── detectors/
│   │   ├── hash_detector.py            # Baseline perceptual hash (aHash, dHash, pHash)
│   │   ├── embedding_detector.py       # Trích xuất đặc trưng Deep CNN zero-mean calibrated
│   │   └── hybrid_detector.py          # Bộ phát hiện đa phương thức & đồ thị tương đồng
│   ├── splitters/
│   │   ├── naive_splitter.py           # Phân chia ngẫu nhiên (gây rò rỉ)
│   │   └── guardian_splitter.py        # Phân chia phân tầng giữ nguyên cụm (Group-Preserving)
│   ├── evaluation/
│   │   ├── leakage_metrics.py          # Tính Primary F1, Recall, Precision, Cross-split Sim
│   │   └── model_trainer.py            # Huấn luyện mô hình so sánh & đánh giá holdout test
│   └── cli/
│       └── split_guardian_gate.py      # Cổng kiểm soát CI/CD tự động
├── benchmarks/
│   ├── run_full_benchmark.py           # Chạy toàn bộ pipeline từ A đến Z
│   └── ablation_study.py               # Thử nghiệm Ablation, threshold sweep & scaling
├── reports/
│   ├── benchmark_summary.json          # Kết quả số liệu thực nghiệm
│   ├── ablation_report.json            # Kết quả ablation & scaling
│   ├── clean_split.json                # Tập split an toàn đã qua kiểm định
│   └── ci_gate_report.json             # Báo cáo quyết định CI/CD gate
├── web_dashboard/
│   ├── index.html                      # Giao diện trực quan hóa tương tác
│   ├── styles.css                      # Giao diện Glassmorphism hiện đại
│   ├── app.js                          # Xử lý biểu đồ động & xem cặp rò rỉ
│   └── data_report.json                # Dữ liệu xuất cho visualizer
├── slides/
│   ├── showcase_slides.html            # Bộ slide 5 trang trình chiếu tương tác
│   └── showcase_slides.md              # Nội dung tóm tắt thuyết trình
└── notebook/
    ├── generate_notebook.py            # Script tạo notebook tự động
    └── c3_leakage_split_guardian.ipynb # Jupyter Notebook hoàn chỉnh có thể chạy ngay
```

---

## 3. Hướng Dẫn Tái Lập Thực Nghiệm (Step-by-step Reproduction)

### Bước 1: Chuẩn bị môi trường
```bash
cd C:\Users\ACER\.gemini\antigravity-ide\scratch\leakage-split-guardian
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ Benchmark kiểm định (One-Command)
Lệnh này sẽ tự động sinh dữ liệu, chạy Baseline, chạy SplitGuardian, kiểm định cặp rò rỉ, huấn luyện mô hình so sánh và xuất báo cáo:
```bash
python benchmarks/run_full_benchmark.py
```

### Bước 3: Chạy Ablation Study & Phân tích Chi phí / Runtime
```bash
python benchmarks/ablation_study.py
```

### Bước 4: Kiểm tra Cổng CI/CD Security Gate
- **Kiểm tra phát hiện và chặn split rò rỉ (Return Code = 1 - Blocked):**
  ```bash
  python src/cli/split_guardian_gate.py --manifest dataset_store/metadata.json --policy configs/guardian_policy.yaml
  ```
- **Kiểm tra thông qua split sạch sau khi re-split (Return Code = 0 - Passed):**
  ```bash
  python src/cli/split_guardian_gate.py --manifest dataset_store/metadata.json --policy configs/guardian_policy.yaml --split reports/clean_split.json
  ```

### Bước 5: Mở Web Dashboard & Slide Showcase
- Khởi chạy web server nội bộ:
  ```bash
  python -m http.server 8080
  ```
- Mở trình duyệt xem Dashboard: `http://localhost:8080/web_dashboard/index.html`
- Mở trình duyệt xem Slide Showcase: `http://localhost:8080/slides/showcase_slides.html`

---

## 4. Dataset, Protocol & License
- **Protocol:** Độc lập hoàn toàn giữa Train/Val và Holdout Test. Scenes 00–13 dành riêng cho huấn luyện và thẩm định rò rỉ; Scenes 14–17 được khóa bảo mật làm Independent Holdout Test Set.
- **Unit of Leakage:** Được định nghĩa ở cả 2 cấp độ:
  - *Pair-level:* Cặp mẫu $(s_i, s_j)$ có quan hệ chuỗi video liên tiếp hoặc chụp đồng thời đa camera.
  - *Cluster-level:* Cụm đồ thị liên thông chứa tất cả các quan sát của một sự kiện vật thể duy nhất.
- **License:** MIT License — Hoàn toàn mở cho mục đích nghiên cứu và tích hợp vào hệ thống MLOps doanh nghiệp.

---

## 5. Ablation Study, Phân Tích Chi Phí & Runtime

Kết quả chạy từ `benchmarks/ablation_study.py`:
- **Modality Ablation:**
  - *Perceptual Hash Only:* F1 = 0.4667 (Bỏ sót đa camera)
  - *Deep Embeddings Only:* F1 = 0.7262 (Nhạy cảm với góc quay lớn)
  - *Metadata Spatiotemporal Only:* F1 = 0.9393
  - *SplitGuardian Hybrid:* F1 = **0.9393**, duy trì 100% Recall trên mọi điều kiện biên.
- **Throughput & Khả năng mở rộng:**
  - Tốc độ xử lý trích xuất và so khớp đạt **~250 - 388 img/s trên CPU** mà không cần GPU chuyên dụng.
  - Phù hợp tích hợp trực tiếp vào worker CI/CD chuẩn của GitHub Actions hoặc GitLab CI.

---

## 6. Interactive Web Dashboard & 5-Slide Showcase
- **Web Dashboard:** Tích hợp bộ duyệt cặp ảnh tương tác (Pair Inspector), biểu đồ trực quan hóa độ tương đồng phân chia, bảng đo lường KPI thời gian thực và giao diện mô phỏng CI/CD Gate.
- **5-Slide Showcase (`slides/showcase_slides.html`):** Thiết kế sẵn dạng slide deck trình chiếu fullscreen với phím mũi tên điều hướng, giải đáp đầy đủ và súc tích toàn bộ 6 câu hỏi cốt lõi của đề tài C3.
