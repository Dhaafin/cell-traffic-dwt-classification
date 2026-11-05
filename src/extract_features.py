"""
Feature Extraction Pipeline: Hybrid 2D-DWT Haar Energies + Physical Time-Domain Statistics.

Extracts 9 features from each 7x24 cellular traffic weekly matrix:
  1. Frequency Domain (2D-DWT Haar Level-1):
     - E_LL: Approximation subband energy (macro weekly traffic load)
     - E_LH: Horizontal detail energy (diurnal hour-to-hour transitions)
     - E_HL: Vertical detail energy (day-to-day variations, weekday vs. weekend)
     - E_HH: Diagonal detail energy (localized bursty traffic anomalies)
  2. Time Domain (Physical Statistics of the 7x24 Matrix):
     - Mean, Median, Std, Max (busy-hour peak), Min (nighttime baseline)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pywt

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "dataset_feature_haar_standar_stats_9.csv"

CLASSES = {
    "UU": "Matrix_UU_User_UL.xlsx",
    "UD": "Matrix_UD_User_DL.xlsx",
    "CU": "Matrix_CU_Cell_UL.xlsx",
    "CD": "Matrix_CD_Cell_DL.xlsx",
}


def extract_haar_energies(image_7x24: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Level-1 2D-DWT Haar subband energies with edge padding (7x24 -> 8x24)."""
    padded_8x24 = np.pad(image_7x24, ((0, 1), (0, 0)), mode="edge")
    LL, (LH, HL, HH) = pywt.dwt2(padded_8x24, "haar")

    e_ll = float(np.mean(LL**2))
    e_lh = float(np.mean(LH**2))
    e_hl = float(np.mean(HL**2))
    e_hh = float(np.mean(HH**2))
    return e_ll, e_lh, e_hl, e_hh


def extract_standard_statistics(
    image_7x24: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute 5 physical summary statistics from the 7x24 matrix."""
    arr = image_7x24.flatten()
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.std(arr)),
        float(np.max(arr)),
        float(np.min(arr)),
    )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    for class_name, file_name in CLASSES.items():
        file_path = RAW_DATA_DIR / file_name
        print(f"Processing class [{class_name}] from: {file_name}...")
        df = pd.read_excel(file_path, sheet_name="Weekly_Data")
        ids = df["ID"].values

        # Impute missing values (NaN) using class-specific hourly means
        pixels_df = df.loc[:, "R0C0":"R6C23"].copy()
        for hour in range(24):
            hour_cols = [
                f"R{day}C{hour}"
                for day in range(7)
                if f"R{day}C{hour}" in pixels_df.columns
            ]
            hour_vals = pixels_df[hour_cols].values.flatten()
            valid_mean = (
                float(np.nanmean(hour_vals)) if np.any(~np.isnan(hour_vals)) else 0.0
            )
            pixels_df[hour_cols] = pixels_df[hour_cols].fillna(valid_mean)

        pixels_matrix = pixels_df.values.astype(float)
        for img_id, row_pixels in zip(ids, pixels_matrix):
            img_7x24 = row_pixels.reshape(7, 24)
            e_ll, e_lh, e_hl, e_hh = extract_haar_energies(img_7x24)
            mean_v, median_v, std_v, max_v, min_v = extract_standard_statistics(img_7x24)

            all_records.append(
                {
                    "ID": img_id,
                    "Class": class_name,
                    "E_LL": e_ll,
                    "E_LH": e_lh,
                    "E_HL": e_hl,
                    "E_HH": e_hh,
                    "Mean": mean_v,
                    "Median": median_v,
                    "Std": std_v,
                    "Max": max_v,
                    "Min": min_v,
                }
            )

    df_features = pd.DataFrame(all_records)
    df_features.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("DONE! Extracted 9 Hybrid Features (2D-DWT Haar + Physical Statistics).")
    print(f"Saved to : {OUTPUT_FILE}")
    print(f"Shape    : {df_features.shape} (1,140 samples x 11 columns)")
    print("=" * 65)


if __name__ == "__main__":
    main()
