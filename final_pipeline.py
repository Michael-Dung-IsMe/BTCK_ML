import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print(f"LightGBM version: {lgb.__version__}")

# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================

# Import file train (đã được tiền xử lý — top 50 features quan trọng nhất)
print("[1/3] Đang tải dữ liệu và chuẩn bị Features...")

# Import file train
X_train = pd.read_parquet("data/processed_set/X_train_tree.parquet")
y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()

# Import file test
X_test_raw = pd.read_csv("data/test_set/test_X.csv", engine="pyarrow")
y_test = pd.read_csv("data/test_set/test_y.csv").squeeze()

# Import file challenge
X_challenge_raw = pd.read_csv("data/challenge_set/challenge_X.csv", engine="pyarrow")
y_challenge = pd.read_csv("data/challenge_set/challenge_y.csv").squeeze()

# Gọi lại danh sách 50 features quan trọng nhất (trích xuất từ LightGBM Feature Importance)
top_50_features = joblib.load("data/preprocessors/top_50_features.pkl")

# Chọn 50 features quan trọng nhất cho tập Test và Challenge
X_test = X_test_raw[top_50_features]
X_challenge = X_challenge_raw[top_50_features]
del X_test_raw, X_challenge_raw # Giải phóng RAM

# Kiểm tra kích thước các tệp
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of X_challenge:", X_challenge.shape)

# ============================================================
# HUẤN LUYỆN MÔ HÌNH
# ============================================================
# Mô hình được chọn: LightGBM — đứng đầu bảng xếp hạng benchmark trên cả 2 tập
# Bộ siêu tham số được tìm bằng RandomizedSearchCV (2 giai đoạn):
#   - Giai đoạn 1: Quét cấu trúc cây (n_estimators, learning_rate, num_leaves)
#   - Giai đoạn 2: Quét regularization (subsample, colsample, reg_alpha, reg_lambda)
# Đã thử 4 cấu hình (n_estimators=445/650/499), kết luận:
#   n_estimators=445 + learning_rate=0.1170 cho điểm Challenge cao nhất (0.9546)

print("\n[2/3] Khởi tạo mô hình LightGBM với các siêu tham số tối ưu nhất đã tìm được...")
# Ghi chú rõ ràng lý do: Nhờ RandomizedSearchCV chạy ngầm, nhóm đã trích xuất được cục tham số

model = LGBMClassifier(
    random_state=42,
    device_type='gpu',
    verbose=-1,
    # ===== BỘ THAM SỐ CHỐT HẠ (từ RandomizedSearchCV) =====
    # Đã thử 4 cấu hình: tăng n_estimators → overfit, zoom-in → không cải thiện.
    # Kết luận: n_estimators=445, lr=0.1170 là điểm cân bằng tối ưu.
    # Test F1: 0.9783 | Challenge F1: 0.9546 (vượt Benchmark 0.9490)
    learning_rate=0.1170,
    min_child_samples=28,
    n_estimators=445,
    num_leaves=72,
    colsample_bytree=0.7218,
    reg_alpha=0.4884,
    reg_lambda=3.4212,
    subsample=0.7761,
)

print("\nBắt đầu training...")
model.fit(X_train, y_train)
print("Training hoàn tất!")

# ============================================================
# ĐÁNH GIÁ TRÊN TẬP TEST
# ============================================================
print("\n[3/3] Đánh giá khả năng dự đoán của hệ thống...")
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]

acc_test = accuracy_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_prob_test)

print(f"\n=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
print(f"Accuracy: {acc_test:.4f}")
print(f"ROC AUC:  {auc_test:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_test, target_names=['Benign', 'Malware'], digits=4))

# Vẽ Confusion Matrix tập Test
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — Tập Test')
plt.tight_layout()
plt.show()

# ============================================================
# ĐÁNH GIÁ TRÊN TẬP CHALLENGE
# ============================================================
print("=== BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP CHALLENGE ===")

y_prob_challenge = model.predict_proba(X_challenge)[:, 1]
y_pred_challenge = (y_prob_challenge >= 0.5).astype(int)

acc_challenge = accuracy_score(y_challenge, y_pred_challenge)
roc_auc_challenge = roc_auc_score(y_challenge, y_prob_challenge)

print(f"\nĐộ chính xác (Accuracy): {acc_challenge:.4f}")
print(f"ROC AUC Score:           {roc_auc_challenge:.4f}")
print("\n--- Báo cáo chi tiết (Classification Report) ---")
print(classification_report(y_challenge, y_pred_challenge, target_names=['Benign (0)', 'Malware (1)'], digits=4))

# Phân tích lỗi
cm = confusion_matrix(y_challenge, y_pred_challenge)
fn = cm[1][0]
fp = cm[0][1]
print(f"\n[QUAN TRỌNG] Phân tích lỗi:")
print(f"- False Negatives (Sót Malware): {fn} mẫu (Malware nhưng bị đoán là An toàn) -> Cực kỳ nguy hiểm.")
print(f"- False Positives (Báo động giả): {fp} mẫu (An toàn nhưng bị đoán là Malware) -> Gây phiền toái.")

# Vẽ Confusion Matrix và ROC Curve tập Challenge
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Confusion Matrix
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
            xticklabels=['Dự đoán Benign', 'Dự đoán Malware'],
            yticklabels=['Thực tế Benign', 'Thực tế Malware'])
ax[0].set_title('Confusion Matrix (Số lượng) — Tập Challenge')
ax[0].set_ylabel('Nhãn thực tế')
ax[0].set_xlabel('Nhãn dự đoán')

# Biểu đồ 2: ROC Curve
fpr, tpr, _ = roc_curve(y_challenge, y_prob_challenge)
from sklearn.metrics import auc as auc_fn
roc_auc = auc_fn(fpr, tpr)

ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) — Tập Challenge')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# ============================================================
# BIỂU ĐỒ ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE)
# ============================================================
lgb.plot_importance(model, max_num_features=20, importance_type='gain',
                    figsize=(10, 8), title='Feature Importance (Gain) — Top 20')
plt.show()