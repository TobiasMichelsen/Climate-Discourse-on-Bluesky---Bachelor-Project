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

# Functions
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

# --- DATA ---

df_whole = pd.DataFrame()
input_path = "../../data/climate_classified"
for filename in os.listdir(input_path):
    df = pd.read_json(f"{input_path}/{filename}")
    df = df[df["label"] == "yes"]
    df = df[df["score"] >= .99]
    df_whole = pd.concat([df_whole, df], ignore_index=True)
    print(df_whole.shape, flush=True)

# Hyperparameters
#Combinations: 2x2x2x2 // Low
embedding_models = ["all-MiniLM-L6-v2"]
min_cluster_sizes = [150, 300]
min_samples_vals = [25, 50]
distance_metrics = ["cosine","manhattan"]
umap_neighbors = [25,50]
umap_components = [7, 9]
umap_min_dist = [0.5]
nr_topics = 15

# Every job gets a separate csv log file:

log_path = os.path.expanduser("logs/bertopic_grid_log_5.csv")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

log_columns = [
    "embedding_model", "metric", "min_cluster_size", "min_samples",
    "nr_topics", "umap_neighbors", "umap_components", "umap_min_dist",
    "n_topics", "outliers", "outlier_pct", "time_sec", "coherence", "diversity"
]

if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
else:
    log_df = pd.DataFrame(columns=log_columns)
    log_df.to_csv(log_path, index=False)

counter_combinations = len(log_df)
max_combinations = 32
proportion = counter_combinations / max_combinations

# --- MAIN LOOP ---
texts_to_embed = df_whole["text"].tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

batch_size = 64

for embed_model in embedding_models:
    print(f"\nStarting embedding with model: {embed_model}", flush=True)
    model = SentenceTransformer(embed_model, device=device)

    start_embed = time.time()
    embeddings_local = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    embed_time = round(time.time() - start_embed, 2)
    print(f"Embedding done for model '{embed_model}' in {embed_time}s", flush=True)

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
            (log_df.embedding_model == run_key["embedding_model"]) &
            (log_df.metric == run_key["metric"]) &
            (log_df.min_cluster_size == run_key["min_cluster_size"]) &
            (log_df.min_samples == run_key["min_samples"]) &
            (log_df.nr_topics == run_key["nr_topics"]) &
            (log_df.umap_neighbors == run_key["umap_neighbors"]) &
            (log_df.umap_components == run_key["umap_components"]) &
            (log_df.umap_min_dist == run_key["umap_min_dist"])
        ]

        if not existing.empty:
            print(f"Skipping already completed: {run_key}", flush=True)
            continue

        print(f"\nRunning: {run_key}", flush=True)
        start = time.time()

        if metric == "cosine":
            embeddings_used = normalize(embeddings_local, norm="l2")
            hdbscan_metric = "euclidean"
        else:
            embeddings_used = embeddings_local
            hdbscan_metric = metric

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
            topic_model.reduce_topics(texts_to_embed, nr_topics=nr_topics)

            topic_info = topic_model.get_topic_info()
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

            print(f"Completed {len(log_df)} models so far.", flush=True)
            print(f"Done | Topics: {n_topics}, Outliers: {n_outliers} ({log_entry['outlier_pct']}%) | "
                  f"Coherence: {coherence:.4f} | Diversity: {diversity:.4f} | Time: {duration}s", flush=True)

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
