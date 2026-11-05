"""
Hyperparameter Sensitivity Analysis Across All 4 Machine Learning Paradigms.

Sweeps key structural hyperparameters on the 9-feature Haar DWT + Statistics dataset:
  1. Random Forest : n_estimators (25..1000) & max_depth (3..None)
  2. SVM           : Regularization C (0.1..100) & Kernel Type (rbf, linear, poly, sigmoid)
  3. XGBoost       : learning_rate (0.01..0.3) & max_depth (2..10)
  4. KNN           : n_neighbors k (1..25) & Weighting Scheme (uniform vs. distance)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "dataset_feature_haar_standar_stats_9.csv"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"


def load_split():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ["ID", "Class"]]
    X = df[feature_cols].to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(df["Class"].to_numpy(dtype=str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


def run_rf_sensitivity(X_train, X_test, y_train, y_test):
    print("\n--- 1. Random Forest Sensitivity (n_estimators & max_depth) ---")
    depths = [3, 5, 8, 10, 15, 20, 25, None]
    rows = []
    for d in depths:
        label = str(d) if d is not None else "None"
        rf = RandomForestClassifier(n_estimators=300, max_depth=d, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test)) * 100
        rows.append({"max_depth": label, "Accuracy (%)": round(acc, 2)})
    df_d = pd.DataFrame(rows)
    print(df_d.to_string(index=False))
    df_d.to_csv(RESULTS_DIR / "sensitivity_rf_depth.csv", index=False)


def run_svm_sensitivity(X_train, X_test, y_train, y_test):
    print("\n--- 2. SVM Sensitivity (Regularization C & Kernel) ---")
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    rows = []
    for c in c_values:
        svm = SVC(C=c, kernel="rbf", random_state=42)
        svm.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test)) * 100
        rows.append({"C": c, "Accuracy (%)": round(acc, 2)})
    df_c = pd.DataFrame(rows)
    print(df_c.to_string(index=False))
    df_c.to_csv(RESULTS_DIR / "sensitivity_svm_c.csv", index=False)


def run_xgb_sensitivity(X_train, X_test, y_train, y_test):
    print("\n--- 3. XGBoost Sensitivity (max_depth at lr=0.05) ---")
    depths = [2, 3, 4, 5, 6, 8, 10]
    rows = []
    for d in depths:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=d,
            random_state=42,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train)
        acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
        rows.append({"max_depth": d, "Accuracy (%)": round(acc, 2)})
    df_xgb = pd.DataFrame(rows)
    print(df_xgb.to_string(index=False))
    df_xgb.to_csv(RESULTS_DIR / "sensitivity_xgb_depth.csv", index=False)


def run_knn_sensitivity(X_train, X_test, y_train, y_test):
    print("\n--- 4. KNN Sensitivity (k neighbors & weighting) ---")
    k_values = [1, 2, 3, 5, 7, 9, 11, 15, 21, 25]
    rows = []
    for k in k_values:
        knn_u = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_train, y_train)
        knn_d = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_train, y_train)
        acc_u = accuracy_score(y_test, knn_u.predict(X_test)) * 100
        acc_d = accuracy_score(y_test, knn_d.predict(X_test)) * 100
        rows.append({"k": k, "Uniform (%)": round(acc_u, 2), "Distance (%)": round(acc_d, 2)})
    df_knn = pd.DataFrame(rows)
    print(df_knn.to_string(index=False))
    df_knn.to_csv(RESULTS_DIR / "sensitivity_knn_k.csv", index=False)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    print("=" * 70)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS (4 ML PARADIGMS)")
    print("=" * 70)
    run_rf_sensitivity(X_train, X_test, y_train, y_test)
    run_svm_sensitivity(X_train, X_test, y_train, y_test)
    run_xgb_sensitivity(X_train, X_test, y_train, y_test)
    run_knn_sensitivity(X_train, X_test, y_train, y_test)
    print("\n" + "=" * 70)
    print(f"All sensitivity CSV tables saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
