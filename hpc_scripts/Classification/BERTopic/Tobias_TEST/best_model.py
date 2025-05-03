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

# --- Load and Filter Data ---
input_path = "../../data/climate_classified"
df_whole = pd.DataFrame()
for filename in os.listdir(input_path):
    df = pd.read_json(f"{input_path}/{filename}")
    df = df[(df["label"] == "yes") & (df["score"] >= 0.99)]
    df_whole = pd.concat([df_whole, df], ignore_index=True)

texts_to_embed = df_whole["text"].tolist()
seqs = df_whole["seq"].tolist()

# --- Set Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Hardcoded Best Hyperparameters ---
embed_model_name = "all-MiniLM-L6-v2"
metric = "cosine"
min_cluster_size = 400
min_samples = 200
n_neighbors = 15
n_components = 7
min_dist = 0.0
nr_topics = 15

# --- Load Embeddings ---
model = SentenceTransformer(embed_model_name, device=device)
embeddings = model.encode(
    texts_to_embed,
    show_progress_bar=True,
    batch_size=64,
    convert_to_numpy=True
)
embeddings = normalize(embeddings, norm="l2")  # Needed for cosine distance

# --- Create UMAP and HDBSCAN models ---
umap_model = UMAP(
    n_neighbors=n_neighbors,
    n_components=n_components,
    min_dist=min_dist,
    metric=metric,
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean",  # cosine metric requires L2-normalization
    cluster_selection_method="eom"
)

# --- Fit BERTopic ---
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=False,
    verbose=True,
    low_memory=True
)

topics, _ = topic_model.fit_transform(texts_to_embed, embeddings)
topic_info = topic_model.get_topic_info()

df_result = pd.DataFrame({
            "seq": seqs,
            "text": texts_to_embed,
            "topic": topics
        })

# --- Save Model ---
save_path = "logs/"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
topic_model.save(save_path)
print(f"Saved BERTopic model to: {save_path}")
df_result.to_json(os.path.join(save_path, "all_clusters.json"), orient= "records", lines=True)


with open(os.path.join(save_path, "topic_info.json"), "w") as f:
    json.dump(topic_info.to_dict(orient="records"), f, indent=2)

# --- Cleanup ---
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
