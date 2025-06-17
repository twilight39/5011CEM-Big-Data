# %%
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
        if pd.notna(row.iloc[3]) and len(str(row.iloc[3])) > 2:
            disease_l1 = row.iloc[3]
            disease_l2 = ""
            disease_l3 = ""
            disease_l4 = ""
        elif pd.notna(row.iloc[4]) and len(str(row.iloc[4])) > 2:
            disease_l2 = row.iloc[4]
            disease_l3 = ""
            disease_l4 = ""
        elif pd.notna(row.iloc[5]) and len(str(row.iloc[5])) > 2:
            disease_l3 = row.iloc[5]
            disease_l4 = ""
        elif pd.notna(row.iloc[6]) and len(str(row.iloc[6])) > 2:
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
