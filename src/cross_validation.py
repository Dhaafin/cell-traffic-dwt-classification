"""
Stratified 5-Fold & 10-Fold Cross-Validation Pipeline (Zero Data Leakage).

Evaluates all 4 tuned models using scikit-learn Pipelines where StandardScaler
is fit strictly inside each training fold. Produces summary statistics (Mean ± Std)
and publication-grade boxplot / bar chart figures.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
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
    y = LabelEncoder().fit_transform(df["Class"].to_numpy(dtype=str))

    models = {
        "Random Forest (depth=8)": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42
        ),
        "SVM (RBF, C=10)": SVC(kernel="rbf", C=10.0, random_state=42),
        "XGBoost (lr=0.05, depth=5)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="mlogloss",
        ),
        "KNN (k=21)": KNeighborsClassifier(n_neighbors=21, weights="uniform"),
    }

    cv_configs = {
        "5-Fold Stratified CV": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "10-Fold Stratified CV": StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        ),
    }

    all_results = []
    fold_records = []

    print("=" * 75)
    print("STRATIFIED CROSS-VALIDATION (5-FOLD & 10-FOLD, ZERO DATA LEAKAGE)")
    print("Dataset: Haar DWT + Standard Statistics (9 Features, N = 1,140)")
    print("=" * 75)

    for cv_name, cv_split in cv_configs.items():
        print(f"\n>>> {cv_name} <<<")
        for model_name, model in models.items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
            scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            scores = cross_validate(
                pipeline, X, y, cv=cv_split, scoring=scoring, n_jobs=-1
            )

            acc_mean = scores["test_accuracy"].mean() * 100
            acc_std = scores["test_accuracy"].std() * 100
            f1_mean = scores["test_f1_macro"].mean()

            for fold_idx, acc_val in enumerate(scores["test_accuracy"] * 100, 1):
                fold_records.append(
                    {
                        "CV Scheme": cv_name,
                        "Model": model_name,
                        "Fold": f"Fold {fold_idx}",
                        "Accuracy": acc_val,
                    }
                )

            all_results.append(
                {
                    "CV Scheme": cv_name,
                    "Model": model_name,
                    "Mean Accuracy (%)": round(acc_mean, 2),
                    "Std Dev (%)": round(acc_std, 2),
                    "Mean ± Std": f"{acc_mean:.2f}% ± {acc_std:.2f}%",
                    "Mean Macro F1": round(f1_mean, 4),
                }
            )
            print(
                f"[{model_name:28}] Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}% | Macro F1: {f1_mean:.4f}"
            )

    df_results = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "hasil_standar_cross_validation.csv"
    df_results.to_csv(out_csv, index=False)

    # Boxplot visualization
    df_folds = pd.DataFrame(fold_records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, (cv_name, palette) in enumerate(
        [("5-Fold Stratified CV", "Blues_r"), ("10-Fold Stratified CV", "Purples_r")]
    ):
        sub_df = df_folds[df_folds["CV Scheme"] == cv_name]
        sns.boxplot(
            data=sub_df,
            x="Model",
            y="Accuracy",
            ax=axes[idx],
            palette=palette,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "red",
                "markersize": 7,
            },
        )
        sns.stripplot(
            data=sub_df,
            x="Model",
            y="Accuracy",
            ax=axes[idx],
            color="black",
            alpha=0.45,
            jitter=0.2,
            size=5.5,
        )
        axes[idx].set_title(f"{cv_name} Distribution", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis="x", rotation=20)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("")
    plt.suptitle(
        "Cross-Validation Accuracy Distribution Across Folds (Red Dot = Mean)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    box_path = ASSETS_DIR / "distribusi_cross_validation_boxplot.png"
    plt.savefig(box_path, dpi=300)
    plt.close()

    print("\n" + "=" * 75)
    print(f"Saved CV summary CSV : {out_csv}")
    print(f"Saved CV Boxplot     : {box_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
