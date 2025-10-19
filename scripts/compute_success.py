import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np

df = pd.read_csv("results/simulated_results.csv")
count = np.sum(df["Q_sim"] > 10) / len(df)

print(f"Proportion of points where Q_sim > 10: {count:.2%}")