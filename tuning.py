import pandas as pd
import numpy as np
import warnings
import time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import uniform, randint

from lightgbm import LGBMClassifier

# --- BernoulliNB (giữ lại từ lần trước) ---
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings('ignore')

# ============================================================
# PHẦN 1: FINE-TUNE LIGHTGBM (RandomizedSearchCV)
# ============================================================
# Chiến lược chia 2 giai đoạn:
#   Giai đoạn 1: Quét cấu trúc cây (n_estimators, learning_rate, num_leaves, min_child_samples)
#   Giai đoạn 2: Quét regularization (subsample, colsample, reg_alpha, reg_lambda)
# Mỗi giai đoạn chạy ~50 cấu hình × 5 folds = 250 lần fit
# ============================================================

def tune_lightgbm():
    print("="*60)
    print("FINE-TUNE LIGHTGBM (2 GIAI ĐOẠN)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Đọc dữ liệu nhánh Tree (50 features)
    print("[1/4] Đang tải dữ liệu nhánh Tree...")
    X_train = pd.read_parquet("data/processed_set/X_train_tree.parquet")
    y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()
    print(f"  -> Shape: {X_train.shape}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)
    
    # ─────────────────────────────────────────────
    # GIAI ĐOẠN 1: Quét cấu trúc cây — THU HẸP quanh vùng tốt nhất đã biết
    # Best so far: learning_rate=0.1170, n_estimators=445, num_leaves=72, min_child_samples=28
    # ─────────────────────────────────────────────
    print("\n[2/4] GIAI ĐOẠN 1: Tinh chỉnh cấu trúc cây (zoom-in)...")
    
    param_dist_phase1 = {
        # n_estimators: 445 là tốt nhất, tăng lên 650 → overfit. Thu hẹp [350, 500]
        'n_estimators': randint(350, 500),
        
        # learning_rate: 0.117 đang tốt. Quét vùng lân cận [0.08, 0.15]
        'learning_rate': uniform(0.08, 0.07),
        
        # num_leaves: 72-75 đang tốt. Quét quanh đó [60, 90]
        'num_leaves': randint(60, 90),
        
        # min_child_samples: 28 đang tốt. Giữ quanh [20, 40]
        'min_child_samples': randint(20, 40),
    }
    
    base_lgbm = LGBMClassifier(
        random_state=42,
        n_jobs=1,       # Không để model tự song song nội bộ
        verbose=-1,
    )
    
    search_phase1 = RandomizedSearchCV(
        estimator=base_lgbm,
        param_distributions=param_dist_phase1,
        n_iter=15,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )
    
    search_phase1.fit(X_train, y_train)
    
    best_phase1 = search_phase1.best_params_
    print(f"\nGiai đoạn 1 XONG!")
    print(f" -> Best params: {best_phase1}")
    print(f" -> Best F1 (CV): {search_phase1.best_score_:.4f}")
    
    # ─────────────────────────────────────────────
    # GIAI ĐOẠN 2: Quét regularization
    # Kế thừa params tốt nhất từ giai đoạn 1
    # ─────────────────────────────────────────────
    print("\n[3/4] GIAI ĐOẠN 2: Quét regularization (subsample, colsample, reg_alpha, reg_lambda)...")
    
    param_dist_phase2 = {
        # subsample: 0.7761 đang tốt. Quét gần đó [0.70, 0.90]
        'subsample': uniform(0.70, 0.20),
        
        # colsample_bytree: 0.7218 đang tốt. Quét gần đó [0.65, 0.85]
        'colsample_bytree': uniform(0.65, 0.20),
        
        # reg_alpha: 0.4884 đang tốt (L1 nhỏ). Quét quanh [0.1, 1.0]
        'reg_alpha': uniform(0.1, 0.9),
        
        # reg_lambda: 3.4212 đang tốt (L2 lớn). Quét quanh [2.0, 5.0]
        'reg_lambda': uniform(2.0, 3.0),
    }
    
    # Tạo model mới KẾ THỪA tham số tốt nhất từ giai đoạn 1
    # Và buộc nó chạy 1 luồng nội bộ để giao CPU cho RandomizedSearchCV bên ngoài
    tuned_lgbm = LGBMClassifier(
        random_state=42,
        n_jobs=1,
        verbose=-1,
        **best_phase1,       # Truyền các tham số cấu trúc tốt nhất
    )
    
    search_phase2 = RandomizedSearchCV(
        estimator=tuned_lgbm,
        param_distributions=param_dist_phase2,
        n_iter=15,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )
    
    search_phase2.fit(X_train, y_train)
    
    best_phase2 = search_phase2.best_params_
    print(f"\n✅ Giai đoạn 2 XONG!")
    print(f"  -> Best regularization: {best_phase2}")
    print(f"  -> Best F1 (CV): {search_phase2.best_score_:.4f}")
    
    # ─────────────────────────────────────────────
    # TỔNG KẾT
    # ─────────────────────────────────────────────
    final_params = {**best_phase1, **best_phase2}
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*60)
    print("TỔNG KẾT FINE-TUNE LIGHTGBM")
    print("="*60)
    print(f"Tổng thời gian: {total_time:.2f} phút")
    print(f"F1-Score tốt nhất (CV 5-Fold): {search_phase2.best_score_:.4f}")
    print(f"TOÀN BỘ THAM SỐ TỐI ƯU (copy vào training.py / evaluate.py):")
    print("-"*60)
    print("LGBMClassifier(")
    print(f"    random_state=42,")
    print(f"    device_type='gpu',")
    print(f"    verbose=-1,")
    for key, value in final_params.items():
        if isinstance(value, float):
            print(f"    {key}={value:.4f},")
        else:
            print(f"    {key}={value},")
    print(")")
    print("-"*60)
    print("[4/4] => Hãy copy cấu hình trên và dán vào training.py & evaluate.py!")


# ============================================================
# PHẦN 2: FINE-TUNE BERNOULLINB (GridSearchCV)
# ============================================================

def tune_bernoulli():
    print("\n" + "="*60)
    print("FINE-TUNE BERNOULLI NAIVE BAYES (GRID SEARCH)")
    print("="*60)
    
    start_time = time.time()
    
    print("[1/3] Đang tải dữ liệu...")
    X_train = pd.read_parquet("data/processed_set/X_train_linear.parquet")
    y_train = pd.read_csv("data/raw_set/train_y.csv").squeeze()
    
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif)),
        ('model', BernoulliNB())
    ])
    
    param_grid = {
        'selector__k': [40, 60, 93],
        'model__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        'model__binarize': [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8],
        'model__fit_prior': [True, False]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("[2/3] Đang chạy GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    total_time = (time.time() - start_time) / 60
    print("\n" + "="*60)
    print("TỔNG KẾT FINE-TUNE BERNOULLINB")
    print("="*60)
    print(f"Tổng thời gian: {total_time:.2f} phút")
    print(f"F1-Score tốt nhất (CV): {grid_search.best_score_:.4f}")
    print(f"Tham số tối ưu: {grid_search.best_params_}")
    print("[3/3] => Hãy copy cấu hình trên và dán vào training.py!")


# ============================================================
# MAIN — Chọn chạy phần nào
# ============================================================
if __name__ == "__main__":
    tune_lightgbm()
    # tune_bernoulli()
