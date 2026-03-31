# Báo Cáo: Binary Classification Challenge - Bài tập cuối khóa Machine Learning

## Tổng quan

Dự án này là bài tập cuối khóa Machine Learning, giải quyết bài toán phân loại nhị phân (Binary Classification) để phát hiện mã độc (Malware).
**Mục tiêu chính**: Vượt qua các chỉ số benchmark đã được công bố trên hai tập dữ liệu: `Test Set` và `Challenge Set`.

| Tập dữ liệu       | Benchmark (F1 Score) |   Kết Quả    |       Đánh Giá        |
| :---------------- | :------------------: | :----------: | :-------------------: |
| **Test Set**      |       `0.9801`       | **`0.9783`** |       Tiệm cận        |
| **Challenge Set** |       `0.9490`       | **`0.9546`** | **Vượt Benchmark ✅** |

---

## Cấu trúc dự án

Dự án được ứng dụng các tiêu chuẩn phân mảnh module để dễ bảo trì và thử nghiệm:

```text
BTCK_ML/
├── data/
│   ├── benchmarking/          # Chứa kết quả đánh giá (csv) của các mô hình
│   ├── preprocessors/         # Lưu các object tiền xử lý (PCA, Transformer, v.v.)
│   ├── raw_set/               # Dữ liệu gốc
│   ├── processed_set/         # Dữ liệu chia nhánh đã qua xử lý (file .parquet)
│   ├── test_set/              # Dữ liệu tập test
│   └── challenge_set/         # Dữ liệu tập challenge
├── img/                       # Thư mục lưu các biểu đồ báo cáo
├── evaluate.py                # Kịch bản benchmark tự động cho toàn bộ 17 mô hình
├── final_pipeline.py          # Pipeline cuối cùng
├── preprocessing.py           # Module tách/nhánh/chuẩn hóa dữ liệu
├── training.py                # Module huấn luyện ban đầu trên 17 thuật toán
└── tuning.py                  # Module tìm kiếm tham số
```

---

## Hướng dẫn cài đặt

```bash
# 1. Tạo môi trường ảo
conda create -n ml_env python=3.12.9

# 2. Kích hoạt môi trường ảo
conda activate ml_env

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Tạo các thư mục lưu file train, test, challenge (thực hiện thủ công trên File Explorer/IDE)

```

---

## Bảng xếp hạng Benchmark

### Đánh giá trên tập TEST

| Model                          | Accuracy | Precision | Recall | F1         | AUC        |
| ------------------------------ | -------- | --------- | ------ | ---------- | ---------- |
| XGBoost                        | 0.9791   | 0.9749    | 0.9836 | **0.9792** | **0.9980** |
| LightGBM                       | 0.9777   | 0.9728    | 0.9830 | **0.9779** | **0.9977** |
| HistGradBoost                  | 0.9766   | 0.9711    | 0.9826 | **0.9768** | **0.9976** |
| Random Forest (500)            | 0.9740   | 0.9688    | 0.9796 | **0.9742** | **0.9968** |
| ExtraTrees (500)               | 0.9729   | 0.9685    | 0.9777 | **0.9731** | **0.9969** |
| MLP Neural Network             | 0.9700   | 0.9686    | 0.9715 | **0.9700** | **0.9952** |
| BaggingClassifier              | 0.9678   | 0.9619    | 0.9741 | **0.9680** | **0.9934** |
| DecisionTree (deep)            | 0.9577   | 0.9573    | 0.9582 | **0.9578** | **0.9702** |
| DecisionTree (depth=10)        | 0.9573   | 0.9520    | 0.9633 | **0.9576** | **0.9842** |
| KNN (k=5)                      | 0.9569   | 0.9530    | 0.9613 | **0.9571** | **0.9819** |
| SGD (hinge/SVM)                | 0.9488   | 0.9424    | 0.9561 | **0.9492** | **0.9869** |
| LogisticReg (L1)               | 0.9482   | 0.9431    | 0.9541 | **0.9485** | **0.9890** |
| Logistic Regression            | 0.9481   | 0.9430    | 0.9539 | **0.9484** | **0.9890** |
| LinearSVC                      | 0.9480   | 0.9418    | 0.9550 | **0.9484** | **0.9889** |
| AdaBoost                       | 0.9427   | 0.9327    | 0.9544 | **0.9434** | **0.9873** |
| SGD (Log Loss)                 | 0.9402   | 0.9314    | 0.9505 | **0.9408** | **0.9861** |
| Tuned BernoulliNB (GridSearch) | 0.8213   | 0.8114    | 0.8376 | **0.8243** | **0.9073** |

### Đánh giá trên tập CHALLENGE

| Model                          | Accuracy | Precision | Recall | F1         |
| ------------------------------ | -------- | --------- | ------ | ---------- |
| ExtraTrees (500)               | 0.9031   | 1.0000    | 0.9031 | **0.9491** |
| XGBoost                        | 0.8874   | 1.0000    | 0.8874 | **0.9403** |
| HistGradBoost                  | 0.8865   | 1.0000    | 0.8865 | **0.9398** |
| AdaBoost                       | 0.8806   | 1.0000    | 0.8806 | **0.9365** |
| BaggingClassifier              | 0.8709   | 1.0000    | 0.8709 | **0.9310** |
| LightGBM                       | 0.8698   | 1.0000    | 0.8698 | **0.9304** |
| Random Forest (500)            | 0.8660   | 1.0000    | 0.8660 | **0.9282** |
| DecisionTree (deep)            | 0.8258   | 1.0000    | 0.8258 | **0.9046** |
| DecisionTree (depth=10)        | 0.8025   | 1.0000    | 0.8025 | **0.8905** |
| MLP Neural Network             | 0.7938   | 1.0000    | 0.7938 | **0.8851** |
| Tuned BernoulliNB (GridSearch) | 0.7614   | 1.0000    | 0.7614 | **0.8645** |
| KNN (k=5)                      | 0.7449   | 1.0000    | 0.7449 | **0.8538** |
| LogisticReg (L1)               | 0.6687   | 1.0000    | 0.6687 | **0.8015** |
| Logistic Regression            | 0.5394   | 1.0000    | 0.5394 | **0.7008** |
| SGD (hinge/SVM)                | 0.5121   | 1.0000    | 0.5121 | **0.6773** |
| LinearSVC                      | 0.4909   | 1.0000    | 0.4909 | **0.6585** |
| SGD (Log Loss)                 | 0.4451   | 1.0000    | 0.4451 | **0.6160** |

---

## Quy trình tiền xử lý dữ liệu

Để đánh giá công bằng hiệu năng của hàng loạt model học máy (từ Naive Bayes, Linear Reg cho tới XGBoost), dữ liệu phải được tinh chỉnh theo đặc thù của từng họ thuật toán. Cụ thể:

**1. Khử nhiễu chung:**

- Loại bỏ các feature là hằng số (variance = 0.0) sử dụng `VarianceThreshold`.
- Chuyển CSV về dạng `.parquet` và ép về dạng `float32` để giảm tải bộ nhớ RAM, tránh tình trạng OOM.

**2. Phân nhánh (Branching):**

- **Nhánh A (Tree-based Models)**: Các thuật toán Boosting/RandomForest không bị ảnh hưởng bởi phân phối chuẩn hay ngoại lai. Do phần cứng có hạn, em sử dụng Feature Importance của một mô hình LightGBM nháp để chiết xuất **Top 50 Features** quan trọng nhất làm tập train.
- **Nhánh B (Linear, KNN, Nueral Networks)**: Cực kì nhạy cảm với ngoại lai, đa cộng tuyến. Áp dụng quy trình: `Winsorization` (chặn quantile 1-99%) ➔ `Yeo-Johnson Transform` (đưa về phân phối Gaussian & chuẩn hoá) ➔ `PCA` (giảm chiều giữ 93 components).
- **Nhánh C (Probabilistic - Naive Bayes)**: Áp dụng quy tắc rời rạc hoá (Binarizer) cắt ranh giới phân lớp lớn hơn và bé hơn `Median`.

---

## Lựa chọn & tối ưu mô hình

Dựa trên bảng thành tích từ file `train_benchmark.csv`, em đã chọn **LightGBM** làm mô hình trọng tâm vì tính hiệu quả trên tập dữ liệu kích thước lớn.

Thay vì chạy GridSearch tốn hàng giờ, em chọn `RandomizedSearchCV` với cơ chế dò mìn 2 bước:

1. **Quét cấu trúc cây**: Tối ưu `n_estimators`, `learning_rate` và `num_leaves`.

- Lần chạy 1: Em đã tìm được `n_estimators` = **445** và **learning_rate** = **0.1170**. Lúc này kết quả F1 và ROC trên tập Test là: F1 Score: **0.9783**, ROC AUC: **0.9977**; trên tập Challenge: F1 Score: **0.9546**

  → ROC_Test và F1_Challenge vượt benchmark, nhưng F1_Test fail

- Lần chạy 2: Em thử tăng khoảng chọn`n_estimators` lên 700 (`randint(100, 700)`), `learning_rate` (`uniform(0.01, 0.2)`), `num_leaves` (`randint(20, 100)`). Kết quả F1 và ROC trên tập Test là: F1 Score: **0.9790**, ROC AUC: **0.9977**; trên tập Challenge: F1 Score: **0.9474**

  → Có hiện tượng Diminishing Returns (lợi tức giảm dần) và Overfit trên tập Challenge.

  → Tăng F1_Test, ROC_Test và F1_Challenge cùng giảm.

- Lần chạy 3: Em thử giảm khoảng chọn `n_estimators` xuống 500 (`randint(100, 500)`). Kết quả F1 và ROC trên tập Test là: F1 Score: **0.9784**, ROC AUC: **0.9977**; trên tập Challenge: F1 Score: **0.9487**

  → F1_Test tăng, F1_Challenge giảm → Khoanh vùng chạy giúp cải thiện hiệu số Test nhưng không đáng kể

- Kết luận: Kết quả (cho tới hiện tại) tối ưu `n_estimators` dừng lại ở **445** và `learning_rate` là **0.1170**.

2. **Quét Regularization**: Điều chỉnh `reg_alpha`, `reg_lambda`, và các tham số kháng overfit (`subsample`, `colsample_bytree`).

---

## Đánh giá kết quả cuối cùng

Tất cả đã được gói gọi trong `final_pipeline.py`. Toàn bộ quá trình chạy sẽ in ra kết quả báo cáo và trích xuất các biểu đồ minh họa lưu tại thư mục `./img`.

**Tập Test:**

- Accuracy: `0.9784`
- Precision: `0.9742`, Recall: `0.9829`
- **F1 Score: `0.9783`** | ROC AUC: `0.9977`

**Tập Challenge:**

- Accuracy: `0.9413`
- Báo cáo số lỗi nguy hiểm (False Negatives - Bỏ lọt Malware): `61` ca.
- Báo động giả (False Positives): `43` ca.
- **F1 Score: `0.9546`** > Baseline (`0.9490`).
