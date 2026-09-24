# Trang 1: Nguy cơ rò rỉ dữ liệu trên nuScenes-mini

**Mục tiêu:** Xác định các cặp ảnh liên quan xuất hiện ở cả Train và Validation trước khi đánh giá YOLOv8n.

## Nội dung trên slide

- **Bài toán:** Phát hiện vật thể 2D bằng YOLOv8n trên ảnh từ 6 camera của nuScenes-mini.
- **Dữ liệu:** 10 scene, 404 sample, tương ứng **2.424 ảnh camera keyframe trước khi lọc** nếu lấy đủ 6 camera cho mỗi sample.
- **Rò rỉ thời gian (temporal leakage):** Các keyframe gần nhau trong cùng scene có thể chứa cùng vật thể và bối cảnh. Keyframe có nhãn cách nhau khoảng 0,5 giây.
- **Rò rỉ đa góc nhìn (multi-view leakage):** Các camera cùng sample có thể quan sát chung vật thể hoặc bối cảnh, nhất là các góc nhìn liền kề.

**Nhãn tham chiếu theo metadata, xét trên từng cặp Train–Validation:**

| Loại cặp | Quy tắc đề xuất |
| --- | --- |
| Nguy cơ theo thời gian (`GT = 1`) | Cùng `scene_token`, khác `sample_token`, khoảng cách thời gian giữa hai sample thỏa `0 < Δt ≤ 1,0 giây` |
| Nguy cơ đa góc nhìn (`GT = 1`) | Cùng `sample_token`, khác camera `channel`, cả hai là keyframe |
| Cặp âm đã kiểm tra (`GT = 0`) | Khác scene, đã kiểm tra metadata hành trình và ảnh để loại các cặp còn trùng bối cảnh |

**Trực quan đề xuất:** Hai ảnh cùng scene tại `t = 0,0 s` và `t ≈ 0,5 s`, lần lượt thuộc Train và Validation. Ghi scene, camera và thời điểm dưới mỗi ảnh.

## Ghi chú thuyết trình

- Đây là **nhãn đại diện theo quy tắc (metadata proxy)**. Cùng sample chưa bảo đảm hai ảnh trùng nội dung thị giác. Các cặp chưa đủ cơ sở gán nhãn giữ trạng thái “chưa xác định”, không tự động tính là âm.
- Ngưỡng 1,0 giây là lựa chọn của thực nghiệm. Dùng `sample.timestamp` để tính `Δt`, đổi microsecond sang giây. Ghép các góc camera bằng `sample_token` vì timestamp từng cảm biến có thể lệch nhẹ.
- Số ảnh thực tế lấy từ manifest sau khi lọc. Với nhãn YOLO 2D, thống nhất cách chiếu bounding box 3D lên ảnh, cắt theo biên ảnh, lọc độ hiển thị và ánh xạ lớp. Box chiếu từ 3D có thể không ôm sát vật thể trên ảnh.
- Nguồn: [nuScenes tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html), [thống kê mini chính thức](https://www.nuscenes.org/tutorials/nuscenes_lidarseg_panoptic_tutorial.html), [schema metadata](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md), [công cụ xuất nhãn 2D](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py).

---

# Trang 2: Bộ dò Split Guardian và chỉ số phát hiện rò rỉ

**Mục tiêu:** Đo khả năng phát hiện cặp ảnh có nguy cơ rò rỉ so với nhãn tham chiếu.

## Nội dung trên slide

1. **Trích xuất đặc trưng:** Dùng DINOv2 hoặc CLIP với trọng số cố định và chuẩn hóa embedding. Chạy pHash làm baseline riêng.
2. **So sánh xuyên tập:** Tính cosine similarity cho các cặp ảnh Train–Validation. Với pHash, dùng khoảng cách Hamming.
3. **Gắn cờ:** Cảnh báo khi cosine similarity `≥ θ`. Lưu hai ảnh, metadata và điểm tương đồng để kiểm tra.
4. **Đánh giá ở mức cặp ảnh:** Báo cáo Precision, Recall và F1 trên tập cặp đã gán nhãn, kèm kết quả riêng cho temporal và multi-view.

| Chỉ số chính | Công thức | Ý nghĩa |
| --- | --- | --- |
| Leakage Precision | `TP / (TP + FP)` | Tỷ lệ cảnh báo khớp nhãn dương tham chiếu |
| Leakage Recall | `TP / (TP + FN)` | Tỷ lệ cặp dương tham chiếu mà bộ dò tìm được |
| Leakage Detection F1 | `2 × P × R / (P + R)` | Cân bằng Precision và Recall |

**Trực quan đề xuất:** Sơ đồ các bước phát hiện và bảng Precision, Recall, F1 theo ngưỡng, điền bằng kết quả thực nghiệm.

## Ghi chú thuyết trình

- Tính `TP`, `FP`, `FN` trên cùng tập cặp tham chiếu. Báo cáo số cặp dương, âm, chưa xác định và cách lấy mẫu. Recall chỉ phản ánh các cặp dương trong phạm vi đánh giá này.
- Chọn backbone và ngưỡng `θ` trên tập hiệu chỉnh, sau đó khóa cấu hình trước khi đánh giá trên các scene khác. Không chọn ngưỡng bằng chính tập báo cáo F1 cuối cùng. pHash có ngưỡng khoảng cách riêng.
- Với mini, có thể xét toàn bộ cặp xuyên tập. Nếu dùng top-k, phải tính cả các cặp dương bị bỏ sót ở bước tìm ứng viên.
- Metadata dùng tạo nhãn tham chiếu và kiểm tra chính sách chia tập. Không đưa chính quy tắc tạo nhãn vào bộ dò embedding rồi coi điểm số đó là bằng chứng nhận diện nội dung ảnh tốt.
- Đồ thị tương đồng dùng ảnh làm đỉnh và cặp bị gắn cờ làm cạnh. Có thể kiểm tra các cụm liên quan trên đồ thị, nhưng không cần clustering để tính chỉ số ở mức cặp.

---

# Trang 3: Chính sách chia theo scene và các đánh đổi

**Mục tiêu:** Ngăn cùng scene xuất hiện ở hai tập và làm rõ ảnh hưởng đến phân bố dữ liệu.

## Nội dung trên slide

| Chính sách | Cách chia | Tác động |
| --- | --- | --- |
| Random split | Chia ngẫu nhiên 80/20 theo từng ảnh | Có nguy cơ đưa các ảnh gần thời gian hoặc cùng sample vào hai tập |
| Scene-level split | Giữ toàn bộ thời gian và 6 camera của mỗi scene trong một tập | Loại bỏ giao thoa scene giữa Train và Validation |

**Ba đánh đổi cần báo cáo:**

| Đánh đổi | Cách theo dõi và xử lý |
| --- | --- |
| Mất cân bằng lớp | Thống kê số box theo lớp và tập. Ghi rõ lớp vắng mặt trong Validation và cách tính mAP khi thiếu lớp |
| Lệch tỷ lệ 80/20 | Báo cáo riêng tỷ lệ scene, ảnh và box. Chia 8/2 scene không bảo đảm đạt 80/20 ở số ảnh hoặc nhãn |
| Biến động theo scene | Lặp lại với các cách chia scene đã định trước, báo cáo độ phân tán và khác biệt bối cảnh |

**Trực quan đề xuất:** Biểu đồ cột số bounding box theo lớp trong Train và Validation của hai chính sách, dùng cùng thứ tự lớp và thang đo.

## Ghi chú thuyết trình

- Điều kiện bắt buộc: `scene_train ∩ scene_val = ∅`. Kiểm tra thêm giao thoa `sample_token` và ảnh trùng hoàn toàn.
- Cân bằng lớp bằng cách di chuyển nguyên scene. Không tách ảnh cùng scene sang hai tập để ép tỷ lệ hoặc bổ sung lớp hiếm.
- Các scene khác nhau vẫn có thể thuộc cùng hành trình hoặc lặp bối cảnh. Scene-level split chưa chứng minh độc lập hoàn toàn về tuyến đường hay địa lý. Nếu mục tiêu là đánh giá trên tuyến đường mới, cần bổ sung quy tắc nhóm theo log hoặc vị trí.
- Nguồn về quan hệ scene và log: [schema nuScenes](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md).

---

# Trang 4: Thiết kế thực nghiệm YOLOv8n và chỉ số bổ sung

**Mục tiêu:** Đánh giá ảnh hưởng của chính sách chia tập và mức tương đồng còn lại.

## Nội dung trên slide

- **Thiết lập chung:** YOLOv8n, cùng checkpoint khởi tạo, cách tạo nhãn, kích thước ảnh, augmentation, batch size, optimizer và learning rate. Huấn luyện 30 epochs mỗi lần chạy.
- **Tính lặp lại:** Dùng cùng danh sách seed cho hai chính sách. Báo cáo trung bình và độ lệch chuẩn, kèm số ảnh và số bước cập nhật.
- **Chênh lệch điểm:** `ΔmAP = mAP_random − mAP_scene`, dùng mAP50–95. Nếu mAP trình bày theo %, chênh lệch có đơn vị **điểm phần trăm**.
- **Tương đồng còn lại:** Với mỗi ảnh Validation, tìm ảnh Train có cosine similarity cao nhất. Báo cáo trung vị, phân vị 95% và tỷ lệ ảnh có điểm này `≥ θ`.

| Kết quả cần đo | Random split | Scene-level split |
| --- | --- | --- |
| mAP50–95, trung bình ± độ lệch chuẩn (%) | Chưa đo | Chưa đo |
| Số scene xuất hiện ở cả hai tập | Chưa đo | Yêu cầu: 0 |
| Trung vị / phân vị 95% của similarity cao nhất mỗi ảnh Validation | Chưa đo | Chưa đo |
| Tỷ lệ ảnh Validation có ít nhất một cảnh báo (%) | Chưa đo | Chưa đo |

**Trực quan đề xuất:** Đường validation mAP theo epoch và phân phối similarity cao nhất mỗi ảnh Validation, trước và sau khi chia lại. Giữ cùng backbone và ngưỡng khi so sánh.

## Ghi chú thuyết trình

- Hai cách chia tạo ra hai tập Validation khác nhau. `ΔmAP` có thể phản ánh cả leakage, độ khó, phân bố lớp và kích thước tập. Không quy toàn bộ chênh lệch cho leakage.
- Để so sánh khả năng tổng quát hóa của hai mô hình, cần đánh giá trên **cùng một tập Test giữ riêng theo scene**, không tham gia huấn luyện, hiệu chỉnh ngưỡng hay chọn mô hình. Nếu trích Test từ mini, áp dụng tỷ lệ Train/Validation trên phần còn lại và công bố số scene thực tế.
- Đổi seed chỉ đo biến động huấn luyện. Để đánh giá độ nhạy với cách chia dữ liệu, lặp scene split như trang 3 và báo cáo riêng nguồn biến động này.
- Với embedding đã chuẩn hóa `z`, đặt `s(v) = max_{t ∈ Train} cosine(z_v, z_t)`. Tỷ lệ cảnh báo là `số ảnh v có s(v) ≥ θ / số ảnh Validation`. Trung bình trên mọi cặp có thể che khuất một số cặp rất giống nhau.
- Không có ngưỡng cosine 0,65 “an toàn” chung cho mọi backbone và dữ liệu. Số cảnh báo giảm cũng chưa đủ chứng minh hai tập độc lập.

---

# Trang 5: Diễn giải kết quả và kiểm tra trước huấn luyện

**Mục tiêu:** Xác định bằng chứng cần có để đánh giá mô hình đáng tin cậy hơn.

## Nội dung trên slide

- **mAP cần đi cùng thông tin chia tập:** Điểm trên random split có thể lạc quan khi Train và Validation chứa các quan sát liên quan.
- **Điểm thấp hơn cần được giải thích:** Scene-level split có thể tạo bài toán khó hơn do bối cảnh chưa thấy hoặc phân bố lớp thay đổi. Điểm giảm chưa đủ kết luận chất lượng mô hình giảm.
- **Bằng chứng cần báo cáo:** Giao thoa metadata, Precision/Recall/F1 của bộ dò và các cặp ảnh bị cảnh báo. Dùng chung Test khi so sánh khả năng tổng quát hóa.
- **Ứng dụng Split Guardian:** Kiểm tra manifest trước huấn luyện, chặn vi phạm quy tắc chia theo scene và chuyển cảnh báo embedding sang bước xem xét ảnh.

**Thông điệp kết thúc:** Độ tin cậy của kết quả phụ thuộc vào ranh giới dữ liệu và cách đánh giá, cùng với điểm mAP.

**Trực quan đề xuất:** Sơ đồ gồm manifest dữ liệu, kiểm tra chia tập, xem xét cảnh báo và huấn luyện. Minh họa một cặp ảnh cảnh báo kèm lý do xử lý.

## Ghi chú thuyết trình

- Lưu phiên bản dataset, cách tạo nhãn, danh sách scene từng tập, seed, cấu hình bộ dò và báo cáo kiểm tra để tái lập thí nghiệm.
- Tích hợp bước kiểm tra vào CI/CD của pipeline dữ liệu. Với chính sách chia theo scene, giao thoa scene là lỗi chặn. Similarity cao là cảnh báo cần kiểm tra, không tự động xóa ảnh chỉ dựa trên điểm embedding.
- nuScenes-mini phù hợp thử nghiệm quy trình. Cần thêm dữ liệu và đánh giá trên bối cảnh giữ riêng trước khi kết luận về hiệu quả triển khai thực tế.
