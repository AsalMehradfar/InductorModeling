import os
import re
import pandas as pd
from tqdm import tqdm
from utils.normalization import compute_normalization_stats, save_normalization_stats, load_normalization_stats
from utils.data_utils import split_dataset

def parse_header_line(line: str):
    """
    Parses the header comment line to extract W, Lv, Lh, Lc.
    """
    pattern = r"W_um=(\d+\.?\d*), Lv_um=(\d+\.?\d*), Lh_um=(\d+\.?\d*), Length_C_um=(\d+\.?\d*)"
    match = re.search(pattern, line)
    if match:
        W, Lv, Lh, Lc = map(float, match.groups())
        return W, Lv, Lh, Lc
    else:
        raise ValueError(f"Header line could not be parsed: {line}")

def load_single_file(path):
    """
    Loads a single file, extracts metadata from the top line, and returns a DataFrame.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Extract values from comment line
    header_line = lines[0].strip()
    if not header_line.startswith("#"):
        raise ValueError("First line does not start with '#'")
    W, Lv, Lh, Lc = parse_header_line(header_line)

    # Read the actual CSV content
    df = pd.read_csv(path, comment="#")

    # Add fixed values to all rows
    df["W_um"] = W
    df["Lv_um"] = Lv
    df["Lh_um"] = Lh
    df["Lc_um"] = Lc

    return df

def clean_data(df):
    """
    Filters the loaded DataFrame to keep only the essential columns.
    Keeps: W_um, Lv_um, Lh_um, Lc_um, freq_GHz, Q, L_pH
    """
    required_cols = ["freq_GHz", "W_um", "L_pH", "Lv_um", "Lh_um", "Lc_um", "Q"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[required_cols]

    df = df[(df["freq_GHz"] < 100) & (df["Q"] > 0) & (df["L_pH"] > 0)]
    df = df.drop_duplicates()
    df.reset_index(inplace=True, drop=True)
    return df

def load_all_data(data_dir="./data/raw/"):
    """
    Loads all CSV files in the data directory and concatenates them into one DataFrame.
    """
    dfs = []
    for fname in tqdm(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            full_path = os.path.join(data_dir, fname)
            try:
                df = load_single_file(full_path)
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Skipping {fname} due to error: {e}")
    if not dfs:
        raise ValueError("No valid CSV files were loaded.")
    
    df_all = pd.concat(dfs, ignore_index=True)

    return clean_data(df_all)

def load_or_process_data(processed_csv_path=None, train_path=None, val_path=None, test_path=None, norm_stats_path=None, feature_cols=None):
    # Case 1: All cached files exist
    if all(os.path.exists(p) for p in [train_path, val_path, test_path, norm_stats_path]):
        print("[INFO] Loading train/val/test and normalization stats from disk...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        mean, std = load_normalization_stats(norm_stats_path)
    else:
        # Process raw data
        print("[INFO] Cached files not found. Processing from scratch...")

        if os.path.exists(processed_csv_path):
            print(f"[INFO] Loading preprocessed data from: {processed_csv_path}")
            df = pd.read_csv(processed_csv_path)
        else:
            print("[INFO] Processed data not found. Running full data loading pipeline...")
            df = load_all_data("data/raw/")
            df.to_csv(processed_csv_path, index=False)
            print(f"[INFO] Processed data saved to: {processed_csv_path}")

        # Split and save
        train_df, val_df, test_df = split_dataset(df)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Compute & save normalization stats
        mean, std = compute_normalization_stats(train_df, feature_cols)
        save_normalization_stats(mean, std, norm_stats_path)

    return train_df, val_df, test_df, mean, std
