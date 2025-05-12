import pandas as pd
import os
from datetime import datetime

#paths
input_path = "../../data/labelled/multi_label/before_merge"
climate_path = "../../data/climate_classified"
save_path = "../../data/labelled/multi_label"

filename_final = "topic_predictions.pkl"
filename_prior = "gpt_topic_predictions.pkl"


#read topic labelled data
df_final = pd.read_pickle(f"{input_path}/{filename_final}")
df_prior = pd.read_pickle(f"{input_path}/{filename_prior}")
print(f"labelled size:\n{df_final.shape}",flush=True)
print(f"labelled size with gpt_labels:\n{df_prior.shape}",flush=True)
#read climate labelled data with repos
df_climate = pd.DataFrame()
for f in os.listdir(climate_path):
    df = pd.read_json(f"{climate_path}/{f}")
    df_climate = pd.concat([df_climate,df])
    print(f"{datetime.now()} loaded: {f}",flush=True)
    
#clean climate data
df_climate = df_climate.drop(["text"],axis=1)
print(f"climate df size:\n{df_climate.shape}",flush=True)
df_climate = df_climate[df_climate["label"] == "yes"]
print(f"after no removal:\n{df_climate.shape}",flush=True)
df_climate = df_climate[df_climate.score >= .99]
print(f"after keep > .99:\n{df_climate.shape}",flush=True)

#merge datasets
print(f"{datetime.now()} merging..",flush=True)
df_join_topic = pd.merge(df_final,df_climate,on="cid",how="left",flush=True)
print(f"after merge full:\n{df_join_topic.shape}",flush=True)
df_join_gpt =  pd.merge(df_final,df_climate,on="cid",how="left")
print(f"after merge gpt:\n{df_join_gpt.shape}",flush=True)

#save merged
print(f"{datetime.now()} saving..",flush=True)
df_join_topic.to_pickle(f"{save_path}/topic_graph_data.pkl")
df_join_gpt.to_pickle(f"{save_path}/gpt_graph_data.pkl")
print(f"{datetime.now()} Finished!",flush=True)
