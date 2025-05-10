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
import kaleido

#start
start_time = datetime.datetime.now()
print("start:",start_time,flush=True)

#CUDA check
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device count:", torch.cuda.device_count(), flush=True)
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA, defaulting to cpu", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load and Filter Data ---
print(f"{datetime.datetime.now()} starting data loading..",flush=True)
input_path = "../../../data/climate_classified"

df_whole = pd.DataFrame()
for filename in os.listdir(input_path):
    df = pd.read_json(f"{input_path}/{filename}")
    df = df[(df["label"] == "yes") & (df["score"] >= 0.99)]
    df_whole = pd.concat([df_whole, df], ignore_index=True)
    print(f"loaded {filename}", flush=True)


df_whole = df_whole[["cid", "text"]]

texts_to_embed = df_whole["text"].tolist()
cids = df_whole["cid"].tolist()

# --- Set Device ---


# --- Hardcoded Best Hyperparameters ---
embed_model_name = "all-MiniLM-L6-v2"
metric = "cosine"
min_cluster_size = 1000
min_samples = 300
n_neighbors = 15
n_components = 7
min_dist = 0.0
nr_topics = 15

# --- Load Embeddings ---
print(f"{datetime.datetime.now()} creating embedding..",flush=True)
model = SentenceTransformer(embed_model_name, device=device)
embeddings = model.encode(
    texts_to_embed,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)
embeddings = normalize(embeddings, norm="l2")  # Needed for cosine distance

# --- Create UMAP and HDBSCAN models ---
print(f"{datetime.datetime.now()} initializing umap..",flush=True)
umap_model = UMAP(
    n_neighbors=n_neighbors,
    n_components=n_components,
    min_dist=min_dist,
    metric=metric,
    random_state=42
)
print(f"{datetime.datetime.now()} initializing hdbscan..",flush=True)
hdbscan_model = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean",  # cosine metric requires L2-normalization
    cluster_selection_method="eom",
    prediction_data=True
)


# --- Fit BERTopic ---
print(f"{datetime.datetime.now()} loading model..",flush=True)
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=False,
    verbose=True,
    low_memory=True
)
print(f"{datetime.datetime.now()} inherence with the model..",flush=True)
topics, _ = topic_model.fit_transform(texts_to_embed, embeddings)
topic_model.reduce_topics(texts_to_embed, nr_topics=nr_topics)
topics = topic_model.transform(texts_to_embed, embeddings=embeddings)[0]
topic_info = topic_model.get_topic_info()

os.makedirs("images", exist_ok=True)
vis_path = "images"

# Topics UMAP projection
fig_topics = topic_model.visualize_topics()

# Customize marker color, size, and layout
fig_topics.update_traces(marker=dict(
    line=dict(width=1, color='DarkSlateGrey')
))

# Update background, title, and remove the range slider
fig_topics.update_layout(
    title="Topic Visualization (run 1)",
    plot_bgcolor='white',
    font=dict(size=16),
    xaxis=dict(rangeslider=dict(visible=False))  # This line removes the slider
)

fig_topics.layout.sliders = []
fig_topics.layout.updatemenus = []

fig_topics.write_image(os.path.join(vis_path, "viz_topics.png"), width=1000, height=800, scale=2)

# Barchart of top words
fig_barchart = topic_model.visualize_barchart(top_n_topics=15)
fig_barchart.write_image(os.path.join(vis_path, "viz_barchart.png"), width=1000, height=800, scale=2)

# Hierarchical clustering
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_image(os.path.join(vis_path, "viz_hierarchy.png"), width=1000, height=800, scale=2)

# Heatmap of topic similarity
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_image(os.path.join(vis_path, "viz_heatmap.png"), width=1000, height=800, scale=2)


df_result = pd.DataFrame({
            "cid": cids,
            "text": texts_to_embed,
            "topic": topics
        })

print("finished!",flush=True)
