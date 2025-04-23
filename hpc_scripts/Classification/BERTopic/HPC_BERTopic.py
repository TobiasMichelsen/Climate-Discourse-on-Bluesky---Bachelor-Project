#import os
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
import csv
import glob



#DATA
df_whole = pd.DataFrame()
input_path = "../../data/climate_classified"
for filename in os.listdir(input_path):
    df = pd.read_json(f"{input_path}/{filename}")
    df = df[df["label"] == "yes"]
    df = df[df["score"] >= .99]
    df_whole = pd.concat([df_whole,df], ignore_index=True)


#HYPERPARAMETERS: EMBEDDING

embedding_models = ["all-MiniLM-L6-v2"]

#HYPERPARAMETERS: HDBSCAN

min_cluster_sizes = [25, 50, 75, 100, 125, 150]

min_samples_vals = [5,10,25,50]

distance_metrics = ["euclidean", "manhattan", "cosine"]

#HYPERPARAMETERS: UMAP

umap_neighbors = [15, 30]

umap_components = [5, 7, 9, 11, 13, 15]

umap_min_dist = [0.0, 0.25]


#TOPIC REDUCTION

nr_topics_vals = [None, 12, 15, 20] 

log_path = os.path.expanduser("logs/bertopic_grid_log.csv")

os.makedirs(os.path.dirname(log_path), exist_ok=True)


#LOGS

log_columns = [
"embedding_model", "metric", "min_cluster_size", "min_samples",
"nr_topics", "umap_neighbors", "umap_components", "umap_min_dist",
"n_topics", "outliers", "outlier_pct", "time_sec"
]

if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
    
else:
    
    log_df = pd.DataFrame(columns=log_columns)
    log_df.to_csv(log_path, index=False)
    
# MAIN LOOP    

texts_to_embed = df_whole["text"].tolist()

for embed_model in embedding_models:
    print(f"\nEmbedding model: {embed_model}")
    model = SentenceTransformer(embed_model)
    embeddings_local = model.encode(texts_to_embed, show_progress_bar=True)


for (min_cluster_size, min_samples, metric, nr_topics,
     n_neighbors, n_components, min_dist) in itertools.product(
    min_cluster_sizes, min_samples_vals, distance_metrics,
    nr_topics_vals, umap_neighbors, umap_components, umap_min_dist
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

    print(f"\nRunning: {run_key}",flush=True)
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
        low_memory=False
    )

    try:
        topics, _ = topic_model.fit_transform(texts_to_embed, embeddings_used)

        df_before = pd.DataFrame({
            "text": texts_to_embed,
            "topic": topics
        })

        topic_model.reduce_topics(texts_to_embed, nr_topics=nr_topics)
        reduced_topics = topic_model.topics_

        df_after = pd.DataFrame({
            "text": texts_to_embed,
            "topic": reduced_topics
        })

        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info[topic_info.Topic != -1])
        n_outliers = topic_info[topic_info.Topic == -1].Count.values[0] if -1 in topic_info.Topic.values else 0
        n_total = sum(topic_info.Count)
        duration = round(time.time() - start, 2)

        log_entry = {
            **run_key,
            "n_topics": n_topics,
            "outliers": n_outliers,
            "outlier_pct": round(n_outliers / n_total * 100, 2),
            "time": duration}

        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        log_df.to_csv(log_path, index=False)

        model_name = f"{embed_model}_{metric}_c{min_cluster_size}_s{min_samples}_nt{nr_topics}_u{n_neighbors}-{n_components}-{min_dist}"
        
        save_dir = f"logs/{model_name}"
        os.makedirs(save_dir, exist_ok=True)

        topic_model.save(os.path.join(save_dir, "model"))
        df_before.to_json(os.path.join(save_dir, "topics_before_reduction.json"), orient= "records", lines=True)
        df_after.to_json(os.path.join(save_dir, "topics_after_reduction.json"), orient= "records", lines=True)

        with open(os.path.join(save_dir, "topic_info.json"), "w") as f:
            json.dump(topic_info.to_dict(orient="records"), f, indent=2)

        print(f"Done | Topics: {n_topics}, Outliers: {n_outliers} ({log_entry['outlier_pct']}%) | Time: {duration}s")

    except Exception as e:
        print(f"Failed for config: {run_key} — {e}")