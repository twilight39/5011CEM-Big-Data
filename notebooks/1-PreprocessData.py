# This script preprocesses mortality data from Malaysia, extracting relevant information from CSV files and preparing it for analysis.

# Known Limitations:
# 1. The script assumes that the CSV files follow a specific naming convention and structure.
# 2. It does not handle cases where the data might be missing or malformed beyond basic error handling.

# %%
# This cell defines the preprocessing steps for the mortality data from Malaysia.
import glob
import os
import re

import pandas as pd

# Constants
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
COUNTRY = "MYS"  # ISO-3 Code for Malaysia


def extract_info_from_filename(filename: str) -> tuple[str, str]:
    """
    Extracts origin year and age group from the filename.

    Args:
        filename (str): The name of the file.

    Returns:
        tuple[str, str]: A tuple containing the year and age group.
    """

    # Example filename: "2020_0-4.csv"
    # This regex captures the year and age group.
    pattern = r"(\d{4})_(.+)\.csv"
    match = re.search(pattern, filename)
    if match:
        year: str = match.group(1)
        age_group: str = match.group(2)
        return year, age_group
    raise ValueError(f"Filename {filename} does not match expected pattern.")


def process_csv_file(file_path: str) -> pd.DataFrame:
    """
    Processes a CSV file and returns the data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """

    year, age_group = extract_info_from_filename(os.path.basename(file_path))

    # Read the file - skip the first 8 rows which contain metadata/headers
    df = pd.read_csv(file_path, skiprows=7, low_memory=False)

    malaysia_col_idx: int = -1
    for col_idx, col_name in enumerate(df.columns):
        if col_name == COUNTRY:
            malaysia_col_idx = col_idx
            break

    if malaysia_col_idx == -1:
        raise ValueError(f"Country '{COUNTRY}' not found in columns of {file_path}")

    result_data: list[dict[str, str | float]] = []

    disease_l1: str = ""
    disease_l2: str = ""
    disease_l3: str = ""
    disease_l4: str = ""

    for index, row in df.iterrows():
        sex: str = row.iloc[0]

        # Extract disease levels
        if pd.notna(row.iloc[3]) and len(str(row.iloc[3])) > 3:
            disease_l1 = row.iloc[3]
            disease_l2 = ""
            disease_l3 = ""
            disease_l4 = ""
        elif pd.notna(row.iloc[4]) and len(str(row.iloc[4])) > 3:
            disease_l2 = row.iloc[4]
            disease_l3 = ""
            disease_l4 = ""
        elif pd.notna(row.iloc[5]) and len(str(row.iloc[5])) > 3:
            disease_l3 = row.iloc[5]
            disease_l4 = ""
        elif pd.notna(row.iloc[6]) and len(str(row.iloc[6])) > 3:
            disease_l4 = row.iloc[6]

        # Skip population rows
        if str(disease_l1).startswith(" Population") or str(disease_l1) == "":
            disease_l1 = ""
            continue

        # Get mortality count for Malaysia
        try:
            mortality_value: str = (
                row.iloc[malaysia_col_idx]
                if pd.notna(row.iloc[malaysia_col_idx])
                else "0"
            )
            # Dataset unit is per 1000 population
            mortality_count = float(mortality_value) * 1000
            mortality_count = int(mortality_count)
        except (ValueError, TypeError) as e:
            print(
                f"Error processing mortality value in file {file_path}, row {index}: {e}"
            )
            mortality_count = 0

        result_data.append(
            {
                "Year": year,
                "Age Group": age_group,
                "Disease_L1": disease_l1,
                "Disease_L2": disease_l2,
                "Disease_L3": disease_l3,
                "Disease_L4": disease_l4,
                "Sex": sex,
                "Mortality Count": mortality_count,
            }
        )

    return pd.DataFrame(result_data)


# %%
# This cell preprocesses and aggregates all CSV files in the raw data directory.
all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
all_data: list[pd.DataFrame] = []

for file in all_files:
    try:
        df = process_csv_file(file)
        all_data.append(df)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

if all_data == []:
    raise ValueError("No data was processed. Check the input files.")

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)

# Save the processed data to a CSV file
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
output_file: str = os.path.join(PROCESSED_DATA_DIR, "malaysia_mortality_data_eda.csv")
final_df.to_csv(output_file, index=False)

# %%
# This cell creates the final model-ready dataframe.

# 1. Define the target levels and features.
FEATURES: list[str] = ["Year", "Age Group", "Sex", "Mortality Count"]

# Arbitrary selection of prediction target that has good balance of detail and number of classes.
TARGET_LEVEL: str = "Disease_L2"
NEXT_LEVEL: str = "Disease_L3"

# Encode 'Age Group' (Ordinal Encoding, because there's a natural order)
AGE_GROUP_MAPPING: dict[str, int] = {
    "0-4": 0,
    "5-14": 1,
    "15-29": 2,
    "30-49": 3,
    "50-59": 4,
    "60-69": 5,
    "70+": 6,
}

# 2. Create a dataframe that ONLY contains rows representing a final L2 category.
# This removes all higher-level aggregates (like L1) and all lower-level details (L3, L4).
model_df = final_df[
    (final_df[TARGET_LEVEL].str.strip() != "")
    & (final_df[NEXT_LEVEL].str.strip() == "")
].copy()


# 3. Create the features DataFrame (X)
features_df = model_df[FEATURES].copy()
features_df["Age Group"] = features_df["Age Group"].map(AGE_GROUP_MAPPING)

# Encode 'Sex' (One-Hot Encoding, because there's no order)
sex_dummies = pd.get_dummies(features_df["Sex"], prefix="Sex")
features_df = pd.concat([features_df, sex_dummies], axis=1)
features_df = features_df.drop("Sex", axis=1)

# 4. Create the target Series (y)
target_series = model_df[TARGET_LEVEL]

# 5. Combine into a single Model DataFrame
features_df.reset_index(drop=True, inplace=True)
target_series.reset_index(drop=True, inplace=True)
final_model_df = pd.concat([features_df, target_series], axis=1)

# 6. Save the model-ready DataFrame to a CSV file
output_model_file: str = os.path.join(
    PROCESSED_DATA_DIR, "malaysia_mortality_data_model.csv"
)
final_model_df.to_csv(output_model_file, index=False)
