"""
Model Training & Evaluation on the 9-Feature Hybrid Representation (Haar DWT + Stats).

Compares Default vs. Tuned Hyperparameter Configurations across 4 ML paradigms:
  - Random Forest (Tuned: n_estimators=300, max_depth=8 -> 88.16%)
  - Support Vector Machine (Tuned: RBF kernel, C=10.0 -> 87.72%)
  - XGBoost (Tuned: n_estimators=300, lr=0.05, max_depth=5 -> 86.84%)
  - K-Nearest Neighbors (Tuned: k=21, weights='uniform' -> 85.09%)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "dataset_feature_haar_standar_stats_9.csv"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
ASSETS_DIR = ROOT_DIR / "assets"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ["ID", "Class"]]
    X = df[feature_cols].to_numpy(dtype=float)

    le = LabelEncoder()
    y = le.fit_transform(df["Class"].to_numpy(dtype=str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    default_models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42, eval_metric="mlogloss"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
    }

    tuned_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="mlogloss",
        ),
        "KNN": KNeighborsClassifier(n_neighbors=21, weights="uniform"),
    }

    print("=" * 80)
    print("STAGE 2: HYPERPARAMETER TUNING EVALUATION (20% INDEPENDENT TEST SPLIT)")
    print("=" * 80)

    summary_rows = []
    cms = {}

    for name in default_models:
        # Evaluate Default
        m_def = default_models[name]
        m_def.fit(X_train_scaled, y_train)
        y_pred_def = m_def.predict(X_test_scaled)
        acc_def = accuracy_score(y_test, y_pred_def) * 100

        # Evaluate Tuned
        m_tun = tuned_models[name]
        m_tun.fit(X_train_scaled, y_train)
        y_pred_tun = m_tun.predict(X_test_scaled)
        acc_tun = accuracy_score(y_test, y_pred_tun) * 100
        prec_tun = precision_score(y_test, y_pred_tun, average="macro")
        rec_tun = recall_score(y_test, y_pred_tun, average="macro")
        f1_tun = f1_score(y_test, y_pred_tun, average="macro")
        delta = acc_tun - acc_def

        cms[name] = (confusion_matrix(y_test, y_pred_tun), acc_tun)

        summary_rows.append(
            {
                "Model": name,
                "Default Accuracy (%)": round(acc_def, 2),
                "Tuned Accuracy (%)": round(acc_tun, 2),
                "Improvement (%)": f"{delta:+.2f}%",
                "Macro Precision": round(prec_tun, 4),
                "Macro Recall": round(rec_tun, 4),
                "Macro F1": round(f1_tun, 4),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    out_csv = RESULTS_DIR / "tuned_models_test_evaluation.csv"
    df_summary.to_csv(out_csv, index=False)

    # Plot 2x2 Confusion Matrix Grid for Optimal Models
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, (name, (cm, acc)) in enumerate(cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{name} (Tuned: {acc:.2f}%)", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Class")
        axes[idx].set_ylabel("Actual Class")

    plt.suptitle(
        "Confusion Matrices of Tuned Models (Haar DWT + Standard Stats 9 Features)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    cm_path = ASSETS_DIR / "heatmap_confusion_matrix_optimal.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Saved evaluation metrics to : {out_csv}")
    print(f"Saved confusion matrix grid : {cm_path}")


if __name__ == "__main__":
    main()
