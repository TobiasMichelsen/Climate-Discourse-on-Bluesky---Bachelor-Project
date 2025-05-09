import os
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt

#start
counter = 90
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

df_yes = df_whole[(df_whole["label"] == "yes")]
df_99_yes = df_whole[(df_whole["label"] == "yes") & (df_whole["score"] >= .99)]
print(f"total:\n {df_whole.shape}",flush=True)
print(f"yes total:\n {df_yes.shape}",flush=True)
print(f"yes .99:\n{df_99_yes.shape}",flush=True)


print("Done!",flush=True)