Mapping Climate Discourse: Building Topic-Centric Social Graphs on Bluesky

Authors: Abel Szemler, Tobias Michelsen
Institution: Data Science, IT University of Copenhagen
Date: 15 May 2025

This project explores climate discourse on Bluesky, a decentralized social media platform. We collected, filtered, and analyzed tens of millions of posts, using machine learning and large language models to map and interpret the structure of climate-related conversations. The result is a topic-labeled, graph-structured dataset and tools for large-scale social media analysis.

Objectivees of the research:

  - Identify dominant topics in Bluesky’s climate discourse

  - Compare Bluesky climate conversations to those on other platforms

  - Evaluate unsupervised clustering + LLM topic labeling at scale

Features: 

  - Data Collection:
    Harvested ~72M posts from Bluesky’s Firehose API, filtered to ~230K high-confidence, climate-related English posts.

  - Climate Classification:
    Uses ClimateBERT, a domain-specific BERT model, to detect climate-relevant texts.

  - Topic Clustering:
    Utilizes BERTopic with MiniLM embeddings, UMAP reduction, and HDBSCAN clustering for scalable topic modeling.

  - Few-Shot LLM Labeling:
    Leverages OpenAI GPT-3.5-turbo for taxonomy-based, human-interpretable topic assignment.

  - Graph Construction:
    Builds topic-centric user graphs: nodes are topics, edges represent user overlap, weighted by topic assignment.

  - Annotation & Evaluation:
    Custom annotation pipeline for validation, using stratified sampling and inter-annotator agreement.


Structure: 

EDA/             # Exploratory Data Analysis scripts and notebooks  
GPT/             # LLM-based topic labeling scripts and outputs  
Visualizations/  # Plots, graphs, and visualization scripts  
data/            # Raw and processed datasets  
documents/       # Project reports, article drafts, and references  
firehose/        # Firehose API data collection and parsing  
hpc_scripts/     # Scripts for high-performance/cluster compute jobs  
notebooks/       # Jupyter notebooks for development and analysis  

