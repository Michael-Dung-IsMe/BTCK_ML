import os
import joblib
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- Import các Model ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import BernoulliNB

# Thư mục chứa bảng điểm chuẩn
BENCHMARK_DIR = "./data/benchmarking"
os.makedirs(BENCHMARK_DIR, exist_ok=True)
TEST_CSV_PATH = f"{BENCHMARK_DIR}/test_benchmark.csv"
CHALLENGE_CSV_PATH = f"{BENCHMARK_DIR}/challenge_benchmark.csv"


def update_benchmark(file_path, new_result: dict):
    """
    Cập nhật hoặc thêm mới kết quả đánh giá mô hình vào file CSV benchmark.
    1. Hàm tự động ghi đè kết quả nếu tên mô hình đã tồn tại, hoặc thêm dòng mới nếu chưa có.
    2. Danh sách được sắp xếp giảm dần theo chỉ số F1 để dễ quan sát.
    
    Args:
        file_path (str): Đường dẫn đến file CSV lưu trữ benchmark.
        new_result (dict): Dictionary chứa kết quả đánh giá, bắt buộc có key 'Model'.
    """
    new_df = pd.DataFrame([new_result])
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Nếu model đã tồn tại thì ghi đè kết quả
        if new_result['Model'] in old_df['Model'].values:
            old_df.set_index('Model', inplace=True)
            new_df.set_index('Model', inplace=True)
            old_df.update(new_df)
            old_df.reset_index(inplace=True)
            final_df = old_df
        else:
            # Nếu chưa có thì concat vào cuối
            final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    
    # Sắp xếp lại danh sách từ cao xuống thấp theo chiều F1
    final_df = final_df.sort_values(by='F1', ascending=False)
    final_df.to_csv(file_path, index=False)


def evaluate_single_model(model, model_name, X_train, y_train, X_test, y_test, X_chal, y_chal):
    """
    Huấn luyện một mô hình cụ thể và đánh giá hiệu năng trên cả 2 tập Test và Challenge.
    Kết quả đo đạc (Accuracy, Precision, Recall, F1, AUC) sẽ được ghi vào 
    các file CSV benchmark tương ứng.
    
    Args:
        model: Đối tượng mô hình học máy (sklearn, lightgbm, v.v.).
        model_name (str): Tên định danh của mô hình.
        X_train, y_train: Dữ liệu huấn luyện.
        X_test, y_test: Dữ liệu kiểm thử nội bộ (Test).
        X_chal, y_chal: Dữ liệu kiểm thử mở rộng (Challenge - phân phối có thể khác biệt).
    """
    
    print(f"\n[{model_name}] Đang fit trên tập Train...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"   -> Fit xong ({time.time() - start_time:.2f}s)")

    # ---------------- 1. Đánh giá trên tập TEST ----------------
    y_test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(X_test)
    else:
        y_test_prob = y_test_pred # Fallback nếu model không hỗ trợ xuất xác suất
        
    test_result = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_test_pred), 4),
        'Precision': round(precision_score(y_test, y_test_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_test_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_test_pred, zero_division=0), 4),
    }
    
    try:
        test_result['AUC'] = round(roc_auc_score(y_test, y_test_prob), 4)
    except ValueError:
        test_result['AUC'] = np.nan
        
    update_benchmark(TEST_CSV_PATH, test_result)
    print(f"   ✓ Test Score     -> F1: {test_result['F1']:.4f} | AUC: {test_result['AUC']:.4f}")
    del y_test_pred, y_test_prob
    
    # ---------------- 2. Đánh giá trên tập CHALLENGE ----------------
    y_chal_pred = model.predict(X_chal)
    if hasattr(model, "predict_proba"):
        y_chal_prob = model.predict_proba(X_chal)[:, 1]
    elif hasattr(model, "decision_function"):
        y_chal_prob = model.decision_function(X_chal)
    else:
        y_chal_prob = y_chal_pred
        
    chal_result = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_chal, y_chal_pred), 4),
        'Precision': round(precision_score(y_chal, y_chal_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_chal, y_chal_pred, zero_division=0), 4),
        'F1': round(f1_score(y_chal, y_chal_pred, zero_division=0), 4),
    }
    
    try:
        chal_result['AUC'] = round(roc_auc_score(y_chal, y_chal_prob), 4)
    except ValueError:
        chal_result['AUC'] = np.nan
        
    update_benchmark(CHALLENGE_CSV_PATH, chal_result)
    print(f"   ✓ Challenge Score-> F1: {chal_result['F1']:.4f} | AUC: {chal_result['AUC']}")


def main():
    print("="*70)
    print(" KHỞI CHẠY QUÁ TRÌNH EVALUATE CÁC MÔ HÌNH TRÊN TEST & CHALLENGE")
    print("="*70)
    
    # =========================================================================
    # TUỲ CHỈNH BẬT/TẮT NHÁNH. HÃY TẮT CÁC NHÁNH CÒN LẠI KHI MÁY THIẾU RAM
    # =========================================================================
    RUN_BRANCH_A = False   # Tree
    RUN_BRANCH_B = True   # Linear / KNN / NN
    RUN_BRANCH_C = False    # Proba (Tuned Bernoulli)
    
    print("\n[INFO] Cấu hình chạy lần này:")
    print(f"- Nhánh A (Tree-based): {RUN_BRANCH_A}")
    print(f"- Nhánh B (Linear):     {RUN_BRANCH_B}")
    print(f"- Nhánh C (Proba):      {RUN_BRANCH_C}")

    # =========================================================================
    # NHÁNH A1: TREE-BASED — CÁC MODEL ĐÃ VƯỢT BENCHMARK
    # Sử dụng tập dữ liệu đã qua rút gọn (Top 50 Features) để tối ưu tốc độ.
    # =========================================================================
    if RUN_BRANCH_A:
        print("\n" + "="*50)
        print(" [NHÁNH A1] TREE-BASED MODELS (Top 50 Features)")
        print("="*50)
        
        X_train = pd.read_parquet("data/processed_set/X_train_tree.parquet")
        y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()
        
        top_50_features = joblib.load("data/preprocessors/top_50_features.pkl")
        
        X_test_raw = pd.read_csv("data/test_set/test_X.csv", engine="pyarrow")
        X_test = X_test_raw[top_50_features]
        y_test = pd.read_csv("data/test_set/test_y.csv").squeeze()
        del X_test_raw
        
        X_chal_raw = pd.read_csv("data/challenge_set/challenge_X.csv", engine="pyarrow")
        X_chal = X_chal_raw[top_50_features]
        y_chal = pd.read_csv("data/challenge_set/challenge_y.csv").squeeze()
        del X_chal_raw
        
        passed_models = {
            "Random Forest (500)": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
            "ExtraTrees (500)": ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1),
            "DecisionTree (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "DecisionTree (deep)": DecisionTreeClassifier(
                max_depth=20, min_samples_split=10, min_samples_leaf=5, random_state=42,
            ),
        }
        
        for name, model in passed_models.items():
            evaluate_single_model(model, name, X_train, y_train, X_test, y_test, X_chal, y_chal)
            
        del X_train, X_test, X_chal
        import gc; gc.collect()

        # =====================================================================
        # NHÁNH A2: CÁC MODEL CHƯA VƯỢT BENCHMARK
        # Lý do: Các mô hình boosting mạnh (LightGBM, XGBoost) biểu diễn yếu nếu 
        #        chỉ dùng 50 features. Chúng cần góc nhìn tổng thể hơn (lọc theo Variance)
        #        để khai thác tối đa quy luật tiềm ẩn.
        # =====================================================================
        print("\n" + "="*50)
        print(" [NHÁNH A2] BOOSTING MODELS (Dùng TOÀN BỘ features sau khi lọc Variance)")
        print("="*50)
        
        # Tải danh sách features có độ phân tán tốt (Variance > 0)
        kept_features = joblib.load("data/preprocessors/kept_features_variance.pkl")
        
        print("[1/4] Đang tải tập Train đầy đủ (parquet → lọc variance)...")
        X_train_full = pd.read_parquet("data/raw_set/train_X.parquet")
        X_train_full = X_train_full[kept_features].astype(np.float32)
        
        print(f"Shape sau variance filter: {X_train_full.shape}")
        
        print("[2/4] Đang tải tập Test đầy đủ (CSV → lọc variance)...")
        X_test_raw = pd.read_csv("data/test_set/test_X.csv", engine="pyarrow")
        X_test_full = X_test_raw[kept_features].astype(np.float32)
        y_test = pd.read_csv("data/test_set/test_y.csv").squeeze()
        del X_test_raw
        
        print("[3/4] Đang tải tập Challenge đầy đủ (CSV → lọc variance)...")
        X_chal_raw = pd.read_csv("data/challenge_set/challenge_X.csv", engine="pyarrow")
        X_chal_full = X_chal_raw[kept_features].astype(np.float32)
        y_chal = pd.read_csv("data/challenge_set/challenge_y.csv").squeeze()
        del X_chal_raw
        
        print("[4/4] Bắt đầu Evaluate 4 model trên toàn bộ features...")
        
        failing_models = {
            "LightGBM": LGBMClassifier(
                random_state=42, device_type='gpu', verbose=-1,
                n_estimators=200,
                num_leaves=128,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.0,
            ),
            "XGBoost": XGBClassifier(
                random_state=42, tree_method='hist', device='cuda',
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
            "HistGradBoost": HistGradientBoostingClassifier(
                random_state=42,
                max_iter=500,
                max_depth=10,
                learning_rate=0.05,
                early_stopping=True,
                min_samples_leaf=20,
            ),
            "BaggingClassifier": BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=15),
                n_estimators=20,
                max_samples=0.8,
                max_features=0.8,
                random_state=42, n_jobs=-1,
            ),
        }
        
        for name, model in failing_models.items():
            evaluate_single_model(model, name, X_train_full, y_train, X_test_full, y_test, X_chal_full, y_chal)
            
        del X_train_full, y_train, X_test_full, y_test, X_chal_full, y_chal
        import gc; gc.collect()

    
    # =========================================================================
    # NHÁNH B: LINEAR / KNN / NN (7 Mô Hình)
    # =========================================================================
    if RUN_BRANCH_B:
        print("\n" + "="*50)
        print(" [NHÁNH B] TIỀN XỬ LÝ DỮ LIỆU & IMPORT (LINEAR/KNN/NN)")
        print("="*50)
        
        X_train = pd.read_parquet("data/processed_set/X_train_linear.parquet")
        y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()
        
        kept_features = joblib.load("./data/preprocessors/kept_features_variance.pkl")
        winsor_bounds = joblib.load("./data/preprocessors/winsorization_bounds.pkl")
        pt_model = joblib.load("./data/preprocessors/power_transformer.pkl")
        pca_model = joblib.load("./data/preprocessors/pca_model.pkl")
        
        def process_linear_branch(df_raw):
            df = df_raw[kept_features]
            df = df.clip(lower=winsor_bounds['lower'], upper=winsor_bounds['upper'], axis=1)
            df_pt = pt_model.transform(df)
            df_pca = pca_model.transform(df_pt)
            return pd.DataFrame(df_pca, columns=[f"pca_comp_{i}" for i in range(1, 94)])
            
        print("Đang chạy Winsorize + Transformer + PCA cho Test/Challenge...")
        X_test_raw = pd.read_csv("data/test_set/test_X.csv", engine="pyarrow")
        X_test = process_linear_branch(X_test_raw)
        y_test = pd.read_csv("data/test_set/test_y.csv").squeeze()
        del X_test_raw
        
        X_chal_raw = pd.read_csv("data/challenge_set/challenge_X.csv", engine="pyarrow")
        X_chal = process_linear_branch(X_chal_raw)
        y_chal = pd.read_csv("data/challenge_set/challenge_y.csv").squeeze()
        del X_chal_raw
        
        linear_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            "LogisticReg (L1)": LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=1000, n_jobs=-1),
            "SGD (Log Loss)": SGDClassifier(loss='log_loss', random_state=42),
            "SGD (hinge/SVM)": SGDClassifier(loss='hinge', random_state=42),
            "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=500, early_stopping=True)
        }
        
        for name, model in linear_models.items():
            evaluate_single_model(model, name, X_train, y_train, X_test, y_test, X_chal, y_chal)
            
        del X_train, y_train, X_test, y_test, X_chal, y_chal
        import gc; gc.collect()


    # =========================================================================
    # NHÁNH C: PROBABILISTIC (Tuned BernoulliNB)
    # =========================================================================
    if RUN_BRANCH_C:
        print("\n" + "="*50)
        print(" [NHÁNH C] TIỀN XỬ LÝ DỮ LIỆU & IMPORT (PROBABILISTIC)")
        print("="*50)
        # Kế thừa Pipeline chuyển đổi toán học từ Nhánh B để chuẩn bị dữ liệu đầu vào
        X_train = pd.read_parquet("data/processed_set/X_train_linear.parquet")
        y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()
        
        kept_features = joblib.load("./data/preprocessors/kept_features_variance.pkl")
        winsor_bounds = joblib.load("./data/preprocessors/winsorization_bounds.pkl")
        pt_model = joblib.load("./data/preprocessors/power_transformer.pkl")
        pca_model = joblib.load("./data/preprocessors/pca_model.pkl")
        
        def process_linear_branch(df_raw):
            df = df_raw[kept_features]
            df = df.clip(lower=winsor_bounds['lower'], upper=winsor_bounds['upper'], axis=1)
            df_pt = pt_model.transform(df)
            df_pca = pca_model.transform(df_pt)
            return pd.DataFrame(df_pca, columns=[f"pca_comp_{i}" for i in range(1, 94)])
            
        print("Đang chạy Transformer pipeline cho Test/Challenge...")
        X_test_raw = pd.read_csv("data/test_set/test_X.csv", engine="pyarrow")
        X_test = process_linear_branch(X_test_raw)
        y_test = pd.read_csv("data/test_set/test_y.csv").squeeze()
        del X_test_raw
        
        X_chal_raw = pd.read_csv("data/challenge_set/challenge_X.csv", engine="pyarrow")
        X_chal = process_linear_branch(X_chal_raw)
        y_chal = pd.read_csv("data/challenge_set/challenge_y.csv").squeeze()
        del X_chal_raw
        
        # Các Best Params được giả lập từ GridSearch
        proba_models = {
            "Tuned BernoulliNB (GridSearch)": Pipeline([
                ('selector', SelectKBest(score_func=f_classif, k=93)),
                ('model', BernoulliNB(alpha=0.1, binarize=0.0, fit_prior=True))
            ])
        }
        
        for name, model in proba_models.items():
            evaluate_single_model(model, name, X_train, y_train, X_test, y_test, X_chal, y_chal)
            
        del X_train, y_train, X_test, y_test, X_chal, y_chal
        import gc; gc.collect()

    print("\n[THÀNH CÔNG] Đã chấm xong vòng lặp và cập nhật kết quả vào ./data/benchmarking/*.csv")


if __name__ == "__main__":
    main()
