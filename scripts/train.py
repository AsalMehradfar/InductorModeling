import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import json
import torch
from pipeline.train_q_predictor import train_q_predictor
from pipeline.data_loader import load_or_process_data
import random
import numpy as np
from utils.metrics import evaluate_model_on_dataframe
from utils.utils import load_model

PROCESSED_CSV_PATH = "data/processed/processed_data.csv"
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
NORM_STATS_PATH = "data/processed/norm_stats.json"


FEATURE_COLS = ["freq_GHz", "W_um", "L_pH", "Lv_um", "Lh_um", "Lc_um"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":

    set_seed(42)
    train_df, val_df, test_df, mean, std = load_or_process_data(PROCESSED_CSV_PATH, 
                                                                TRAIN_PATH, 
                                                                VAL_PATH, 
                                                                TEST_PATH, 
                                                                NORM_STATS_PATH,
                                                                FEATURE_COLS)

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    best_model_path = train_q_predictor(train_df, val_df, mean, std)

    print("Evaluating model on test set...")
    # best_model_path = "checkpoints/q_predictor_best.pt"
    model = load_model(best_model_path)
    test_metrics = evaluate_model_on_dataframe(model, test_df, mean, std, input_columns=FEATURE_COLS)
    
    output_path = f"results/test_metrics.json"
    with open(output_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

