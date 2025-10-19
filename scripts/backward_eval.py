import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils.normalization import load_normalization_stats 
from pipeline.backward_optimize import optimize_lv_lh_lc
from utils.utils import load_model
import pandas as pd
from tqdm import tqdm

FEATURE_COLS = ["freq_GHz", "W_um", "L_pH", "Lv_um", "Lh_um", "Lc_um"]


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--freq", type=float, required=True, help="Frequency in GHz")
    # parser.add_argument("--W", type=float, required=True, help="W in micro meters")
    # parser.add_argument("--L", type=float, required=True, help="L in pico Henries")
    parser.add_argument("--data_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/q_predictor_best.pt")
    parser.add_argument("--stats_path", type=str, default="data/processed/norm_stats.json")
    args = parser.parse_args()

    device = "cpu"

    # load data
    df = pd.read_csv(args.data_path)[:5000]

    # Load stats
    mean, std = load_normalization_stats(args.stats_path)

    # Load model
    model = load_model(args.model_path)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Optimizing..."):
        result = optimize_lv_lh_lc(
            model=model,
            freq=row["freq_GHz"],
            W=row["W_um"],
            L=row["L_pH"],
            mean=mean,
            std=std,
            input_columns=FEATURE_COLS,
            device=device
        )
        # Store combined result
        results.append({
            "freq": row["freq_GHz"],
            "W": row["W_um"],
            "L": row["L_pH"],
            "Lv_pred": round(result["Lv"], 2),
            "Lh_pred": round(result["Lh"], 2),
            "Lc_pred": round(result["Lc"], 2),
            "Q_pred": result["Predicted_Q"]
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv("results/optimized_results.csv", index=False)

    print("✅ Saved to results/optimized_results.csv")


if __name__ == "__main__":
    main()