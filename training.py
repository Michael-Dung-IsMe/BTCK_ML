import pandas as pd
import numpy as np
import warnings
import time
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Tree-based ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier

# --- Linear / Distance / Neural ---
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# --- Probabilistic ---
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH METRICS & HÀM ĐÁNH GIÁ
# ============================================================

scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall'   : make_scorer(recall_score),
    'f1'       : make_scorer(f1_score),
    # response_method: thử predict_proba trước (Tree, LogReg, MLP...)
    # nếu không có thì fallback sang decision_function (LinearSVC, SGD hinge)
    # Thay thế needs_threshold=True đã bị deprecated từ sklearn >= 1.4
    'auc'      : make_scorer(roc_auc_score, response_method=('predict_proba', 'decision_function')),
}

# Dictionary để lưu kết quả nhằm xuất ra CSV
results_list = []

def evaluate_model(model, X, y, model_name="Model", n_jobs=-1):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        # n_jobs=-1: chạy 5 fold song song — dùng cho dataset nhỏ (Tree, Linear)
        # n_jobs=1 : chạy tuần tự — bắt buộc khi dataset lớn (Proba: 2394 features)
        # vì Windows dùng process-based backend (loky), mỗi worker copy toàn bộ
        # dữ liệu sang subprocess riêng → 5 processes × 2.36 GB → OOM → NaN
        n_jobs=n_jobs,
        return_train_score=True, # phát hiện overfitting (train score >> test score => overfit)
    )

    # Tính trung bình các score
    mean_scores = {
        'Model': model_name,
        'Accuracy': scores['test_accuracy'].mean().round(4),
        'Precision': scores['test_precision'].mean().round(4),
        'Recall': scores['test_recall'].mean().round(4),
        'F1': scores['test_f1'].mean().round(4),
        'AUC': scores['test_auc'].mean().round(4),
        # 'Train_F1': scores['train_f1'].mean().round(4) # Thêm để theo dõi overfit trong CSV
    }
    results_list.append(mean_scores)
    

    print(f"\n--- Kết quả CV (5-Fold) cho {model_name} ---")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        test_key  = f'test_{metric}'
        train_key = f'train_{metric}'
        test_mean  = scores[test_key].mean()
        train_mean = scores[train_key].mean()
        overfit_flag = " ⚠ overfit?" if (train_mean - test_mean) > 0.03 else ""
        print(f"  {metric.upper():9s}: val={test_mean:.4f}  train={train_mean:.4f}{overfit_flag}")
    print("-" * 50)
    return mean_scores


start_time = time.time()

# Import file y_train
y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()

# ============================================================
# NHÁNH A — TREE-BASED (X_train_tree: top-50 features gốc)
# ============================================================
print("\n" + "="*60)
print("NHÁNH A: TREE-BASED MODELS")
print("="*60)

X_train_tree = pd.read_parquet("data/processed_set/X_train_tree.parquet")

# 1. LightGBM
lgbm_model = LGBMClassifier(
    random_state=42,
    device_type='gpu',   # Tận dụng GPU (GTX 1660Ti)
    n_estimators=100,
    verbose=-1,          # Tắt log spam của LGBM
)
evaluate_model(lgbm_model, X_train_tree, y_train, "LightGBM")

# 2. XGBoost
# Bỏ eval_metric — không có tác dụng trong CV (chỉ dùng khi có eval_set)
xgb_model = XGBClassifier(
    random_state=42,
    tree_method='hist',
    device='cuda',       # Tận dụng GPU
)
evaluate_model(xgb_model, X_train_tree, y_train, "XGBoost")

# 3. Random Forest — tăng lên 500 cây cho đúng cấu hình benchmark
rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)
evaluate_model(rf_model, X_train_tree, y_train, "Random Forest (500)")

# 4. HistGradientBoosting
hist_gb = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=200,
    early_stopping=True,  # Tránh underfit / overfit tự động
)
evaluate_model(hist_gb, X_train_tree, y_train, "HistGradBoost")

# 5. Bagging Classifier — giới hạn max_depth để tránh cây con quá sâu
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    random_state=42,
    n_jobs=-1,
)
evaluate_model(bagging, X_train_tree, y_train, "BaggingClassifier")

# 6. Extra Trees — tăng lên 500 cây cho đúng cấu hình benchmark
extra_trees = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)
evaluate_model(extra_trees, X_train_tree, y_train, "ExtraTrees (500)")

# 7. Decision Tree (depth=10)
dt_depth10 = DecisionTreeClassifier(max_depth=10, random_state=42)
evaluate_model(dt_depth10, X_train_tree, y_train, "DecisionTree (depth=10)")

# 8. Decision Tree (deep — không giới hạn depth, sẽ overfit hoàn toàn)
dt_deep = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_deep, X_train_tree, y_train, "DecisionTree (deep)")

# 9. AdaBoost (mặc định dùng DecisionTree max_depth=1 — đúng chuẩn)
adaboost = AdaBoostClassifier(random_state=42)
evaluate_model(adaboost, X_train_tree, y_train, "AdaBoost")

# Giải phóng RAM sau khi xong nhánh Tree
del X_train_tree

# Lưu kết quả vào file CSV (ghi đè lên dữ liệu file trước đó)
# benchmark = pd.read_csv("./data/benchmarking/train_benchmark.csv")
# benchmark = pd.DataFrame(results_list).sort_values(by='F1', ascending=False)
# benchmark.to_csv("./data/benchmarking/train_benchmark.csv", index=False)


# ============================================================
# NHÁNH B — LINEAR / KNN / NEURAL NETWORK
# (X_train_linear: Winsorize → Yeo-Johnson → PCA 93 components)
# ============================================================
print("\n" + "="*60)
print("NHÁNH B: LINEAR / KNN / MLP MODELS")
print("="*60)

X_train_linear = pd.read_parquet("data/processed_set/X_train_linear.parquet")

# 1. Logistic Regression (L2 — mặc định)
log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000,
    n_jobs=-1,
)
evaluate_model(log_reg, X_train_linear, y_train, "Logistic Regression")

# 2. Logistic Regression (L1 — saga solver bắt buộc để hỗ trợ L1)
log_reg_l1 = LogisticRegression(
    penalty='l1',
    solver='saga',
    random_state=42,
    max_iter=1000,
    n_jobs=-1,
)
evaluate_model(log_reg_l1, X_train_linear, y_train, "LogisticReg (L1)")

# 3. SGD (log_loss) — tương đương Logistic Regression online
# SGDClassifier KHÔNG hỗ trợ n_jobs — đã xóa
sgd_log = SGDClassifier(
    loss='log_loss',
    random_state=42,
)
evaluate_model(sgd_log, X_train_linear, y_train, "SGD (Log Loss)")

# 4. SGD (hinge) — tương đương Linear SVM online
# SGDClassifier KHÔNG hỗ trợ n_jobs — đã xóa
sgd_hinge = SGDClassifier(
    loss='hinge',
    random_state=42,
)
evaluate_model(sgd_hinge, X_train_linear, y_train, "SGD (hinge/SVM)")

# 5. LinearSVC — AUC tính qua decision_function (không phải xác suất)
linear_svc = LinearSVC(
    random_state=42,
    max_iter=2000,  # Tăng để tránh convergence warning
)
evaluate_model(linear_svc, X_train_linear, y_train, "LinearSVC")

# 6. KNN (k=5)
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1,
)
evaluate_model(knn_model, X_train_linear, y_train, "KNN (k=5)")

# 7. MLP Neural Network (256 → 128)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    random_state=42,
    max_iter=500,         # Tăng từ 300 lên 500 để đảm bảo hội tụ
    early_stopping=True,  # Ngăn overfitting tự động
)
evaluate_model(mlp_model, X_train_linear, y_train, "MLP Neural Network")

# Giải phóng RAM sau khi xong nhánh Linear
del X_train_linear

# Lưu kết quả vào file CSV
# benchmark = pd.read_csv("./data/benchmarking/train_benchmark.csv")
# results_df = pd.DataFrame(results_list).sort_values(by='F1', ascending=False)
# benchmark = pd.concat([benchmark, results_df], ignore_index=True)
# benchmark.to_csv("./data/benchmarking/train_benchmark.csv", index=False)

# ============================================================
# NHÁNH C — PROBABILISTIC ĐƯỢC GRID SEARCH TỐI ƯU
# ============================================================
print("\n" + "="*60)
print("NHÁNH C: PROBABILISTIC MODEL (BERNOULLINB + GRID SEARCH)")
print("="*60)
# Lấy lại tập X_train_linear để thuật toán có dữ liệu cực mịn
X_train_proba = pd.read_parquet("data/processed_set/X_train_linear.parquet")
# 1. Pipeline tích hợp lọc nhiễu + model
bnb_pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif)),
    ('model', BernoulliNB())
])
# 2. Định nghĩa cấu hình quét lưới (Mình thu nhỏ lưới Grid lại một chút để chạy dưới 2 phút lúc chấm điểm)
param_grid = {
    'selector__k': [40, 93],
    'model__alpha': [0.1, 1.0, 5.0],
    'model__binarize': [-0.2, 0.0, 0.2],
    'model__fit_prior': [True, False]
}
# 3. Yêu cầu 6: Tối ưu Hyperparameter
print("Đang tiến hành dò tìm Parameters tốt nhất bằng GridSearchCV (5-folds)...")
grid_search = GridSearchCV(
    estimator=bnb_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1 # In ra 1 dòng ngắn gọn tiến trình (tránh bị rác màn hình)
)
grid_search.fit(X_train_proba, y_train)
# 4. Yêu cầu 5: Đánh giá bằng Cross Validation trên cấu hình Vàng đã tìm được
best_model = grid_search.best_estimator_
print(f"-> Đã tìm ra tham số tối ưu: {grid_search.best_params_}")
# Hàm evaluate_model cũ của bạn sẽ chấm điểm bộ Pipeline tốt nhất này
evaluate_model(best_model, X_train_proba, y_train, "Tuned BernoulliNB (GridSearch)", n_jobs=-1)
del X_train_proba

# Lưu kết quả vào file CSV
benchmark = pd.read_csv("./data/benchmarking/train_benchmark.csv")
results_df = pd.DataFrame(results_list).sort_values(by='F1', ascending=False)
benchmark = pd.concat([benchmark, results_df], ignore_index=True)
benchmark.to_csv("./data/benchmarking/train_benchmark.csv", index=False)

total_time = (time.time() - start_time) / 60
print("\n" + "="*60)
print("--- Hoàn thành đánh giá tất cả mô hình! ---")
print(f"Kết quả đã được lưu tại: ./data/benchmarking/train_benchmark.csv")
print(f"Tổng thời gian thực thi: {total_time:.2f} phút")
print("="*60)