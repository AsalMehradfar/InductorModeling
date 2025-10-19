import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/processed_data.csv")  # or wherever your full dataset lives

mean_q = df["Q"].mean()
std_q = df["Q"].std()

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Q"], 
    bins=50, 
    kde=False, 
    stat="density", 
    color='#2a4d69', 
    edgecolor='#4d4d4d', 
    linewidth=0.5
)

plt.xlabel("Quality Factor (Q)", labelpad=10, fontweight='bold')
plt.ylabel("Density", labelpad=10, fontweight='bold')
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

stats_text = f"μ = {mean_q:.2f}\nσ = {std_q:.2f}"
plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes,
         fontsize=13,
         linespacing=1.6,
         ha='right', va='top',
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2a4d69", linewidth=0.8))

plt.tight_layout()
plt.savefig("results/fig_q_distribution.png", bbox_inches="tight", dpi=400)
plt.show()