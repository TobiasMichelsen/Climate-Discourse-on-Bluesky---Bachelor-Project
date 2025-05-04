# Adapted BERTopic clustering script for second iteration (Run2)
import json
import time
import itertools
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.preprocessing import normalize
import os
import torch
import gc

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# --- Functions ---
def compute_coherence_score(topic_model, documents, top_n_words=10):
    topics = topic_model.get_topics()
    topic_words = [
        [word for word, _ in topics[topic_id][:top_n_words]]
        for topic_id in topics.keys() if topic_id != -1
    ]
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()

def compute_topic_diversity(topic_model, top_n_words=10):
    topics = topic_model.get_topics()
    topic_words = [
        [word for word, _ in topics[topic_id][:top_n_words]]
        for topic_id in topics.keys() if topic_id != -1
    ]
    all_words = [word for words in topic_words for word in words]
    unique_words = len(set(all_words))
    total_words = len(all_words)
    return unique_words / total_words if total_words > 0 else 0


# Load clustered data from previous run1

base_dir = "/home/absz/BachProj/Classification"
cluster_input_path = os.path.join(base_dir, "run1/logs/all_clusters.json")
df_clustered = pd.read_json(cluster_input_path, lines=True)

# Filter to only include documents from cluster 0

df_whole = df_clustered[df_clustered["topic"] == 0].copy()
print(f"Loaded {len(df_whole)} documents from topic 0", flush=True)


# --- Hyperparameters ---

#Combinations: Only uses 1 for testing

embedding_models = ["all-MiniLM-L6-v2"]
min_cluster_sizes = [25,50]
min_samples_vals = [10,20]
distance_metrics = ["cosine"]
umap_neighbors = [10,20]
umap_components = [7,9]
umap_min_dist = [0.0,0.3]
nr_topics = 15

# --- CUDA Check ---
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device count:", torch.cuda.device_count(), flush=True)
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA, defaulting to cpu", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Setup ---
log_path = os.path.expanduser("run2/logs/bertopic_grid_log_2_3.csv")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

log_columns = [
    "embedding_model", "metric", "min_cluster_size", "min_samples",
    "nr_topics", "umap_neighbors", "umap_components", "umap_min_dist",
    "n_topics", "outliers", "outlier_pct", "time_sec", "coherence", "diversity"
]

log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=log_columns)
log_df.to_csv(log_path, index=False)


counter_combinations = len(log_df)
max_combinations = (
    len(min_cluster_sizes)
    * len(min_samples_vals)
    * len(distance_metrics)
    * len(umap_neighbors)
    * len(umap_components)
    * len(umap_min_dist)
)
proportion = counter_combinations / max_combinations

# --- Embedding ---
texts_to_embed = df_whole["text"].tolist()
seqs = df_whole["seq"].tolist()

batch_size = 64

for embed_model in embedding_models:
    model = SentenceTransformer(embed_model, device=device)
    embeddings_local = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )

    for (min_cluster_size, min_samples, metric,
         n_neighbors, n_components, min_dist) in itertools.product(
        min_cluster_sizes, min_samples_vals, distance_metrics,
        umap_neighbors, umap_components, umap_min_dist
    ):

        run_key = {
            "embedding_model": embed_model,
            "metric": metric,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "nr_topics": nr_topics,
            "umap_neighbors": n_neighbors,
            "umap_components": n_components,
            "umap_min_dist": min_dist
        }

        existing = log_df[
            (log_df[list(run_key)] == pd.Series(run_key)).all(axis=1)
        ]

        if not existing.empty:
            print(f"Skipping already completed: {run_key}", flush=True)
            continue

        print(f"\nRunning: {run_key}", flush=True)
        start = time.time()

        embeddings_used = normalize(embeddings_local, norm="l2") if metric == "cosine" else embeddings_local
        hdbscan_metric = "euclidean" if metric == "cosine" else metric

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
            metric=hdbscan_metric,
            cluster_selection_method="eom"
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            language="english",
            calculate_probabilities=False,
            verbose=False,
            low_memory=True
        )

        try:
            topics, _ = topic_model.fit_transform(texts_to_embed, embeddings_used)
            topic_info = topic_model.get_topic_info()

            # Create unique filename based on current hyperparameters
            
            filename = (
                f"topic_info_mcs{min_cluster_size}_ms{min_samples}_nn{n_neighbors}"
                f"_nc{n_components}_dist{min_dist}.json"
            )
            save_path = "run2/logs"
            os.makedirs(save_path, exist_ok=True)
            
            topic_info_path = os.path.join(save_path, filename)

            # Save as JSON
            with open(topic_info_path, "w") as f:
                json.dump(topic_info.to_dict(orient="records"), f, indent=2)

            
            
            df_result = pd.DataFrame({
            "seq": seqs,
            "text": texts_to_embed,
            "topic": topics
            })
            
        
            os.makedirs("run2", exist_ok=True)
            df_result.to_json("run2/logs/clusters_run2_3.json", orient="records", lines=True)

            # Metric calculation
            n_topics = len(topic_info[topic_info.Topic != -1])
            n_outliers = topic_info[topic_info.Topic == -1].Count.values[0] if -1 in topic_info.Topic.values else 0
            n_total = sum(topic_info.Count)
            duration = round(time.time() - start, 2)

            coherence = compute_coherence_score(topic_model, texts_to_embed)
            diversity = compute_topic_diversity(topic_model)

            log_entry = {
                **run_key,
                "n_topics": n_topics,
                "outliers": n_outliers,
                "outlier_pct": round(n_outliers / n_total * 100, 2),
                "time_sec": duration,
                "coherence": round(coherence, 4),
                "diversity": round(diversity, 4)
            }

            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
            log_df.to_csv(log_path, index=False)

            print(f"Done | Topics: {n_topics}, Outliers: {n_outliers} ({log_entry['outlier_pct']}%) |"
                  f" Coherence: {coherence:.4f} | Diversity: {diversity:.4f} | Time: {duration}s", flush=True)
            
            counter_combinations += 1
            proportion = counter_combinations / max_combinations

            if max_combinations >= 10 and counter_combinations % (max_combinations // 10) == 0:
                print(f"Progress: {counter_combinations}/{max_combinations} models completed "
                      f"({int(proportion * 100)}%)", flush=True)

        except Exception as e:
            print(f"Failed for config: {run_key} — {e}", flush=True)

        finally:
            del topic_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
