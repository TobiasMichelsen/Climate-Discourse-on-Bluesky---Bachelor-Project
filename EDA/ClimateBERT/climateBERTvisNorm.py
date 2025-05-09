import numpy as np
import os
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt

#start
start_time = datetime.datetime.now()
print("start:",start_time,flush=True)

# --- Load and Filter Data ---
print(f"{datetime.datetime.now()} starting data loading..",flush=True)
input_path = "../../data/climate_classified"
df_whole = pd.DataFrame()
for filename in os.listdir(input_path):
    df = pd.read_json(f"{input_path}/{filename}")
    df = df[(df["label"] == "yes")]
    df_whole = pd.concat([df_whole, df], ignore_index=True)
    print(f"loaded {filename}", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

# Flatten the 2D array of axes for easier indexing
axes = axes.flatten()


# Calculate the maximum frequency across all histograms
hist_data = [
    df_whole["score"],
    df_whole[df_whole["score"] > 0.7]["score"],
    df_whole[df_whole["score"] > 0.9]["score"],
    df_whole[df_whole["score"] > 0.99]["score"],
]

# Use numpy to compute histograms without plotting

max_freq = 0
for data in hist_data:
    counts, _ = np.histogram(data, bins=100)
    max_freq = max(max_freq, counts.max())

# Then in your plotting section
for ax, data, title, color in zip(
    axes,
    hist_data,
    [
        "all yes label",
        "yes label with scores > 0.7",
        "yes label with scores > 0.9",
        "yes label with scores > 0.99",
    ],
    ["green", "red", "lightgreen", "red"],
):
    ax.hist(data, bins=100, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, max_freq)  # Normalize y-axis
    ax.grid(True)

plt.tight_layout()
plt.savefig("score_histogram_normalized.png", dpi=300)
plt.show()
