"""
Stage 1 Benchmark: Evaluates All 7 Feature Engineering & Ablation Scenarios (S1 to S7).

Reproduces Table 1 of the research paper across 4 Machine Learning paradigms:
  - S1: Baseline Haar 2D-DWT (4 features)
  - S2: Daubechies db2 2D-DWT (4 features)
  - S3: Haar + Subband Statistics [Skewness/Kurtosis] (16 features)
  - S4: db2 + Subband Statistics [Skewness/Kurtosis] (16 features)
  - S5: Ablation Study - Pure Time-Domain Statistics (5 features)
  - S6: Daubechies db2 + Time-Domain Statistics (9 features)
  - S7: Haar 2D-DWT + Time-Domain Statistics (9 features - Optimal)
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

SCENARIOS = {
    "S1: Baseline Haar Energi (4 Feat)": "dataset_feature_haar.csv",
    "S2: Daubechies db2 Energi (4 Feat)": "dataset_feature_db2.csv",
    "S3: Haar + Subband Stats (16 Feat)": "dataset_feature_haar_stats_16.csv",
    "S4: db2 + Subband Stats (16 Feat)": "dataset_feature_db2_stats_16.csv",
    "S5: Pure Time-Domain Stats (5 Feat)": "dataset_feature_pure_statistics_5.csv",
    "S6: db2 + Standard Stats (9 Feat)": "dataset_feature_db2_standar_stats_9.csv",
    "S7: Haar + Standard Stats (9 Feat)": "dataset_feature_haar_standar_stats_9.csv",
}


def get_baseline_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, random_state=42, eval_metric="mlogloss"
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print("=" * 85)
    print("STAGE 1 BENCHMARK: FEATURE ENGINEERING & ABLATION STUDY (7 SCENARIOS x 4 MODELS)")
    print("=" * 85)

    for scenario_name, csv_name in SCENARIOS.items():
        csv_path = PROCESSED_DIR / csv_name
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ["ID", "Class"]]
        X = df[feature_cols].to_numpy(dtype=float)
        y = LabelEncoder().fit_transform(df["Class"].to_numpy(dtype=str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        row = {"Scenario": scenario_name, "Features": len(feature_cols)}
        for m_name, model in get_baseline_models().items():
            model.fit(X_train_s, y_train)
            acc = accuracy_score(y_test, model.predict(X_test_s)) * 100
            row[m_name] = f"{acc:.2f}%"
        records.append(row)

    df_table = pd.DataFrame(records)
    print(df_table.to_string(index=False))
    print("=" * 85)

    out_csv = RESULTS_DIR / "stage1_feature_scenarios_benchmark.csv"
    df_table.to_csv(out_csv, index=False)
    print(f"Saved full benchmark table to: {out_csv}")


if __name__ == "__main__":
    main()
