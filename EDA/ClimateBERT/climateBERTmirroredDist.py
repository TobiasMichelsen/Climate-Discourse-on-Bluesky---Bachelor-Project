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
    df_whole = pd.concat([df_whole, df], ignore_index=True)
    print(f"loaded {filename}", flush=True)

# Separate 'no' and 'yes' labels
df_no = df_whole[df_whole["label"] == "no"].copy()
df_yes = df_whole[df_whole["label"] == "yes"].copy()

# Multiply 'no' scores by -1
df_no["score"] = 1 - df_no["score"]

# Combine them back
df_combined = pd.concat([df_no, df_yes], ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(df_yes["score"], bins=100, alpha=0.6, label="Yes", color='green', edgecolor='black')
plt.hist(df_no["score"], bins=100, alpha=0.6, label="No (mirrored)", color='red', edgecolor='black')

plt.title("Mirrored Score Histogram: Yes (Blue) vs No (Red)")
plt.xlabel("Score")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mirrored_score_histogram.png", dpi=300)
plt.show()