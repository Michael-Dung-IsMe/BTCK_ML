import os
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# =============================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================
RAW_DIR = "./data/raw_set"
PROCESSED_DIR = "./data/processed_set"

X_TRAIN_PARQUET = f"{RAW_DIR}/train_X.parquet"
X_TRAIN_CSV = f"{RAW_DIR}/train_X.csv"
Y_TRAIN_CSV = f"{RAW_DIR}/train_y.csv"

# ============================================
# XỬ LÝ DỮ LIỆU
# ============================================

def preprocess():
    """
    Tiền xử lý toàn bộ dữ liệu huấn luyện và tạo các bộ dữ liệu nhánh cho từng loại mô hình.
    
    Quy trình:
    1. Import dữ liệu: Đọc file train (CSV/Parquet) và ép kiểu float32 để tối ưu hóa bộ nhớ RAM.
    2. Feature Selection: Dùng VarianceThreshold loại bỏ đặc trưng hằng số (không có sự phân tán).
    3. Phân nhánh dữ liệu:
       - Nhánh A (Tree-based): Rút trích Top 50 features quan trọng nhất dưạ trên mô hình LightGBM.
       - Nhánh B (Linear/KNN/NN): Xử lý ngoại lai (Winsorize) -> Biến đổi phân phối chuẩn (Yeo-Johnson) -> Nén chiều (PCA).
       - Nhánh C (Probabilistic): Rời rạc hóa (Binarization) dữ liệu xung quanh ngưỡng Median.
    4. Lưu trữ pipeline: Trích xuất các preprocessors sang file .pkl để dùng lại cho tập Test/Challenge.
    """

    start_time = time.time()

    print("="*60)
    print("GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU & TẠO NHÁNH")
    print("="*60)

    # 1. IMPORT DỮ LIỆU
    print("\n[1/3] Đang tải dữ liệu và ép kiểu float32...")
    if os.path.exists(X_TRAIN_PARQUET):
        X_train = pd.read_parquet(X_TRAIN_PARQUET)
        print(f"  -> Đã đọc từ: {X_TRAIN_PARQUET}")
    elif os.path.exists(X_TRAIN_CSV):
        X_train = pd.read_csv(X_TRAIN_CSV, engine="pyarrow")
        print(f"  -> Đã đọc từ: {X_TRAIN_CSV}")
    else:
        raise FileNotFoundError("Không tìm thấy file dữ liệu X_train!")

    y_train = pd.read_csv(Y_TRAIN_CSV, engine="pyarrow").squeeze()

    X_train = X_train.astype(np.float32)
    print(f"  -> Kích thước X_train ban đầu: {X_train.shape}")
    
    # 2. FEATURE SELECTION: Loại bỏ các cột hằng số với VarianceThreshold
    print("\n[2/3] Chạy VarianceThreshold(0.0) để lọc các cột không có sự biến thiên (hằng số)...")
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)
    
    kept_features = X_train.columns[selector.get_support()]
    removed_cols_count = X_train.shape[1] - len(kept_features)
    X_train = X_train[kept_features]
    print(f" -> Đã loại bỏ {removed_cols_count} cột. Kích thước còn lại: {X_train.shape}")
    
    # Lưu danh sách đặc trưng được giữ lại vào file .pkl
    preprocessors_dir = f"./data/preprocessors"
    os.makedirs(preprocessors_dir, exist_ok=True)
    joblib.dump(kept_features, f"{preprocessors_dir}/kept_features_variance.pkl")


    # 3. TẠO CÁC TẬP DỮ LIỆU NHÁNH (DATA BRANCHING)
    print("\n[3/3] XỬ LÝ DỮ LIỆU ĐẶC THÙ CHO TỪNG NHÓM MÔ HÌNH...\n")
    
    # ==========================================
    # NHÁNH A: MÔ HÌNH TREE-BASED
    # Đặc điểm: Boosting tree không cần chuẩn hóa phân phối, nhưng cần một lượng biến
    #           vừa đủ để học tốt mà không bị bóp nghẽn thời gian tính toán do dư thừa.
    # ==========================================
    print("=== Phân nhánh A: Tree-based Models ===")
    print(" -> Lựa chọn Top 50 features quan trọng nhất dựa trên LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    
    feature_importances = pd.Series(lgb_model.feature_importances_, index=X_train.columns)
    top_50_features = feature_importances.sort_values(ascending=False).head(50).index
    
    X_train_tree = X_train[top_50_features].copy()
    
    joblib.dump(top_50_features, f"{preprocessors_dir}/top_50_features.pkl")
    
    tree_path = f"{PROCESSED_DIR}/X_train_tree.parquet"
    X_train_tree.to_parquet(tree_path)
    print(f" -> Đã cắt {len(top_50_features)} cột và lưu tại: {tree_path} (Shape: {X_train_tree.shape})")
    
    del X_train_tree, lgb_model, feature_importances

    # ==========================================
    # NHÁNH B: MÔ HÌNH LINEAR / KNN / NEURAL NETWORK
    # Đặc điểm: Bị ảnh hưởng mạnh bởi outlier và đa cộng tuyến (multicollinearity).
    # Xử lý:  Cắt trị số kỳ dị -> Cân bằng chuẩn hóa -> PCA giảm chiều đặc trưng.
    # ==========================================
    print("\n=== Phân nhánh B: Linear, KNN, Neural Network ===")
    X_train_linear = X_train.copy()
    
    # Bước 1: Winsorization - Chặn/Cắt các giá trị ngoại lai ở ngưỡng 1% và 99%
    print(" -> Bước 1/3: Winsorization (cắt các điểm kỳ dị 1% và 99%)...")
    lower_bounds = X_train_linear.quantile(0.01)
    upper_bounds = X_train_linear.quantile(0.99)
    X_train_linear = X_train_linear.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    joblib.dump({'lower': lower_bounds, 'upper': upper_bounds}, f"{preprocessors_dir}/winsorization_bounds.pkl")

    # Bước 2: PowerTransformer - Ép phân phối dữ liệu về dạng chuẩn (Gaussian)
    print(" -> Bước 2/3: PowerTransformer (biến đổi Yeo-Johnson & Standardize)...")
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_linear_pt = power_transformer.fit_transform(X_train_linear)
    X_train_linear = pd.DataFrame(X_train_linear_pt, columns=X_train_linear.columns, index=X_train_linear.index)
    joblib.dump(power_transformer, f"{preprocessors_dir}/power_transformer.pkl")

    # Bước 3: PCA - Giảm thiểu đa cộng tuyến, giữ 93 feature có tính đóng góp cao
    print(" -> Bước 3/3: Huấn luyện PCA giữ lại 93 components...")
    pca = PCA(n_components=93, random_state=42)
    X_train_linear_pca = pca.fit_transform(X_train_linear)
    
    X_train_linear = pd.DataFrame(
        X_train_linear_pca, 
        columns=[f"pca_comp_{i}" for i in range(1, 94)], 
        index=X_train_linear.index
    ).astype(np.float32) # Tiếp tục ép float32 cho dữ liệu nén
    joblib.dump(pca, f"{preprocessors_dir}/pca_model.pkl")

    linear_path = f"{PROCESSED_DIR}/X_train_linear.parquet"
    X_train_linear.to_parquet(linear_path)
    print(f" -> Đã lưu tập Linear/KNN/NN tại: {linear_path} (Shape: {X_train_linear.shape})")
    
    # Giải phóng bộ nhớ
    del X_train_linear, X_train_linear_pt, X_train_linear_pca

    # ==========================================
    # NHÁNH C: MÔ HÌNH PROBABILISTIC (BernoulliNB)
    # Đặc điểm: Yêu cầu thuộc tính ở dạng boolean/binary (độc lập có/không).
    # ==========================================
    print("\n=== Phân nhánh C: Probabilistic (BernoulliNB) ===")
    print(" -> Binarization: Phân tách đặc trưng thành 0/1 dựa trên ngưỡng Median...")
    X_train_proba = X_train.copy()
    
    medians = X_train_proba.median().values
    X_train_proba = (X_train_proba > medians).astype(np.float32)
    joblib.dump(medians, f"{preprocessors_dir}/binarizer_medians.pkl")
    
    proba_path = f"{PROCESSED_DIR}/X_train_proba.parquet"
    X_train_proba.to_parquet(proba_path)
    print(f" -> Đã lưu nhánh Probabilistic tại: {proba_path} (Shape: {X_train_proba.shape})")
    
    del X_train_proba

    # ==========================================
    total_time = (time.time() - start_time) / 60
    print("\n" + "="*60)
    print(f"HOÀN THÀNH GIAI ĐOẠN 1! TẤT CẢ FILE & PARAMETERS ĐÃ ĐƯỢC LƯU.")
    print(f"Tổng thời gian thực thi: {total_time:.2f} phút")
    print("="*60)


def visualize_pipeline_results():
    """
    Trích xuất một tập mẫu ngẫu nhiên (5000 samples) để trực quan hóa (EDA)
    và đánh giá tác động của quy trình tiền xử lý lên phân phối dữ liệu.
    """
    print("\n" + "="*60)
    print("VẼ BIỂU ĐỒ SO SÁNH TRƯỚC VÀ SAU KHI TIỀN XỬ LÝ")
    print("="*60)
    
    img_dir = "./img"
    os.makedirs(img_dir, exist_ok=True)
    
    print("[1/6] Đang tải dữ liệu gốc và Preprocessors (từ joblib)...")
    if os.path.exists(X_TRAIN_PARQUET):
        X_train_raw = pd.read_parquet(X_TRAIN_PARQUET)
    else:
        X_train_raw = pd.read_csv(X_TRAIN_CSV, engine="pyarrow")
    y_train = pd.read_csv(Y_TRAIN_CSV, engine="pyarrow").squeeze()
    
    # --- BIỂU ĐỒ 0: THỐNG KÊ NHÃN (CLASS BALANCE) ---
    print("[2/6] Đang vẽ Biểu đồ Phân phối Nhãn (Class Balance)...")
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=y_train, palette="Set1")
    total = len(y_train)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.02 * total,
                    f'{height}\n({height/total*100:.1f}%)', 
                    ha="center", fontsize=11, fontweight='bold')
    plt.title("Phân phối nhãn (Target / Class Balance)", fontsize=14, fontweight='bold')
    plt.xlabel("Nhãn (Target)", fontsize=12)
    plt.ylabel("Số lượng (Count)", fontsize=12)
    plt.ylim(0, y_train.value_counts().max() * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/target_distribution.png', dpi=300)
    plt.close()
    
    preprocessors_dir = "./data/preprocessors"
    kept_features = joblib.load(f"{preprocessors_dir}/kept_features_variance.pkl")
    top_50_features = joblib.load(f"{preprocessors_dir}/top_50_features.pkl")
    winsor_bounds = joblib.load(f"{preprocessors_dir}/winsorization_bounds.pkl")
    pt_model = joblib.load(f"{preprocessors_dir}/power_transformer.pkl")
    pca_model = joblib.load(f"{preprocessors_dir}/pca_model.pkl")
    medians = joblib.load(f"{preprocessors_dir}/binarizer_medians.pkl")
    
    top_3_cols = list(top_50_features[:3])
    X_train_kept = X_train_raw[kept_features].copy()
    
    print("[3/6] Trích xuất ngẫu nhiên 5000 mẫu để sinh ảnh nhanh & không bị OOM...")
    sample_indices = np.random.choice(X_train_kept.index, size=min(5000, len(X_train_kept)), replace=False)
    X_sample = X_train_kept.loc[sample_indices].copy()
    y_sample = y_train.loc[sample_indices].copy()
    
    # Render lại kết quả Linear Pipeline cho các mẫu này
    X_sample_winsor = X_sample.clip(lower=winsor_bounds['lower'], upper=winsor_bounds['upper'], axis=1)
    X_sample_pt = pd.DataFrame(pt_model.transform(X_sample_winsor), columns=X_sample.columns, index=X_sample.index)
    X_sample_pca = pd.DataFrame(pca_model.transform(X_sample_pt), columns=[f"PCA_{i}" for i in range(1, 94)], index=X_sample.index)
    
    print("[4/6] Đang vẽ Biểu đồ Phân phối Feature (Trước vs Sau khi chuẩn hóa)...")
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    fig.suptitle('So sánh Phân phối của Top 3 Features\n(Trước và Sau khi Winsorization + Yeo-Johnson Transform)', fontsize=16, fontweight='bold')

    for i, col in enumerate(top_3_cols):
        sns.histplot(X_sample[col], bins=50, kde=True, ax=axes[i, 0], color='salmon')
        axes[i, 0].set_title(f"Feature '{col}' - Gốc (Raw)")
        sns.histplot(X_sample_pt[col], bins=50, kde=True, ax=axes[i, 1], color='teal')
        axes[i, 1].set_title(f"Feature '{col}' - Sau Standardize + Yeo-Johnson")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{img_dir}/dist_before_after_transform.png')
    plt.close()
    
    print("[5/6] Đang vẽ Scatter 3D So sánh Không gian Dữ liệu: Gốc vs PCA...")
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    targets = y_sample.unique()
    colors = ['blue', 'red']
    
    for target, color in zip(targets, colors):
        subset = X_sample[y_sample == target]
        ax1.scatter(subset[top_3_cols[0]], subset[top_3_cols[1]], subset[top_3_cols[2]], 
                    c=color, label=f'Class {target}', alpha=0.5, s=15)
    ax1.set_xlabel(f"Feature {top_3_cols[0]}")
    ax1.set_ylabel(f"Feature {top_3_cols[1]}")
    ax1.set_zlabel(f"Feature {top_3_cols[2]}")
    ax1.set_title("Không gian 3D: Top 3 Features Gốc")
    if len(targets) > 0: ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    for target, color in zip(targets, colors):
        subset = X_sample_pca[y_sample == target]
        ax2.scatter(subset['PCA_1'], subset['PCA_2'], subset['PCA_3'], 
                    c=color, label=f'Class {target}', alpha=0.5, s=15)
    ax2.set_xlabel('PCA_1')
    ax2.set_ylabel('PCA_2')
    ax2.set_zlabel('PCA_3')
    ax2.set_title("Không gian 3D: Top 3 Thành phần chính PCA")
    if len(targets) > 0: ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{img_dir}/scatter_3d_raw_vs_pca.png')
    plt.close()

    print("[6/6] Đang vẽ Tác động của Binarizer theo Median (Probabilistic Branch)...")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    feat = top_3_cols[0]
    sns.histplot(X_sample[feat], bins=50, kde=True, ax=axes[0], color='gray')
    
    idx = list(kept_features).index(feat) if feat in kept_features else 0
    med_val = medians[idx]
    
    axes[0].axvline(med_val, color='r', linestyle='--', label=f'Median = {med_val:.2f}')
    axes[0].set_title(f"Feature '{feat}' Gốc")
    axes[0].legend()
    
    binarized = (X_sample[feat] > med_val).astype(int)
    sns.countplot(x=binarized, ax=axes[1], palette='Set2')
    axes[1].set_title(f"Sau khi biến đổi nhị phân (Threshold = Median)")
    plt.tight_layout()
    plt.savefig(f'{img_dir}/binarizer_impact.png')
    plt.close()

    print(f"-> HOÀN TẤT VẼ BIỂU ĐỒ! Tất cả ảnh đã được lưu tại {img_dir}/")


if __name__ == "__main__":
    # preprocess()
    visualize_pipeline_results()

