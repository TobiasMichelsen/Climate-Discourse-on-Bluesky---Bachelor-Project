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
df_whole = df_whole[df_whole.label == "yes"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

# Flatten the 2D array of axes for easier indexing
axes = axes.flatten()

# Plot 1: Scores > 0.9
axes[0].hist(df_whole["score"], bins=100, color='green', edgecolor='black')
axes[0].set_title("all yes label")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

# Plot 2: Scores > 0.7
axes[1].hist(df_whole[df_whole["score"] > 0.7]["score"], bins=100, color='red', edgecolor='black')
axes[1].set_title("yes label with scores > 0.7")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

# Plot 3: Scores > 0.9
axes[2].hist(df_whole[df_whole["score"] > 0.9]["score"], bins=100, color='lightgreen', edgecolor='black')
axes[2].set_title("yes label with scores > 0.9")
axes[2].set_xlabel("Score")
axes[2].set_ylabel("Frequency")
axes[2].grid(True)

# Plot 4: Scores < 0.7
axes[3].hist(df_whole[df_whole["score"] > 0.99]["score"], bins=100, color='red', edgecolor='black')
axes[3].set_title("yes label with scores > 0.99")
axes[3].set_xlabel("Score")
axes[3].set_ylabel("Frequency")
axes[3].grid(True)

plt.tight_layout()
plt.savefig("score_histogram.png", dpi=300)  # Save the figure to a file
plt.show()