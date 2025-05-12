import os
import pandas as pd
import torch
import gc
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
import json
import datetime

#start
start_time = datetime.datetime.now()
print("start:",start_time)

#CUDA check
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device count:", torch.cuda.device_count(), flush=True)
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA, defaulting to cpu", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load and Filter Data ---
print(f"{datetime.datetime.now()} starting data loading..")
input_path = "../data/run1_results"
filename = "run1_result.json"

df = pd.read_json(f"{input_path}/{filename}", lines=True)
df = df[df.topic == 0]
df_whole = df[["cid", "text"]]
print(f"loaded {filename}",flush=True)

texts_to_embed = df_whole["text"].tolist()
cids = df_whole["cid"].tolist()


# --- Hardcoded Best Hyperparameters ---
embed_model_name = "all-MiniLM-L6-v2"
metric = "cosine"
min_cluster_size = 200
min_samples = 50
n_neighbors = 20
n_components = 7
min_dist = 0.0
nr_topics = 50

# --- Load Embeddings ---
print(f"{datetime.datetime.now()} creating embedding..")
model = SentenceTransformer(embed_model_name, device=device)
embeddings = model.encode(
    texts_to_embed,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)
embeddings = normalize(embeddings, norm="l2")  # Needed for cosine distance

# --- Create UMAP and HDBSCAN models ---
print(f"{datetime.datetime.now()} initializing umap..")
umap_model = UMAP(
    n_neighbors=n_neighbors,
    n_components=n_components,
    min_dist=min_dist,
    metric=metric,
    random_state=42
)
print(f"{datetime.datetime.now()} initializing hdbscan..")
hdbscan_model = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean",  # cosine metric requires L2-normalization
    cluster_selection_method="eom",
    prediction_data=True
)


# --- Fit BERTopic ---
print(f"{datetime.datetime.now()} loading model..")
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=False,
    verbose=True,
    low_memory=True
)
print(f"{datetime.datetime.now()} inherence with the model..")
topics, _ = topic_model.fit_transform(texts_to_embed, embeddings)
topic_info = topic_model.get_topic_info()

df_result = pd.DataFrame({
            "cid": cids,
            "text": texts_to_embed,
            "topic": topics
        })

# --- Save Model ---
print(datetime.datetime.now(),"saving model..")
save_path = "../data/run2_results"
os.makedirs(save_path, exist_ok=True)
print(f"Saved BERTopic model to: {save_path}")
df_result.to_json(os.path.join(save_path, "run2_result.json"), orient= "records", lines=True)

with open(os.path.join(save_path, "topic_info.json"), "w") as f:
    json.dump(topic_info.to_dict(orient="records"), f, indent=2)

print("finished!")
