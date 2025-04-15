from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import load_dataset
import json
import os
import datetime
import pandas as pd

FILTERED_FOLDER = "../../data/filtered/filtered_test"
OUTPUT_FOLDER = "../../data/climate_classified"
dataset_name = "climatebert/climate_detection"
model_name = "climatebert/distilroberta-base-climate-detector"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
os.makedirs(OUTPUT_FOLDER, exist_ok = True)
files = os.listdir(FILTERED_FOLDER)
while True:
    for filename in files:
        processing_start_time = datetime.datetime.now()
        number = filename.split("_")[-1].replace(".json", "")
        out_filename = f"climate_classified_posts_{number}.json"
        
        #Check if already processed
        if os.path.exists(f"{OUTPUT_FOLDER}/{out_filename}"):
            print(f"Skipping {filename} (already processed)")
            continue
        
        print(f"processing: {filename}\n")
        file_path = f"{FILTERED_FOLDER}/{filename}"
        dataset = load_dataset("json", data_files=file_path, split="train")
        # dataset = dataset[:100]
        print(type(dataset))
        df = pd.DataFrame(dataset)
        print(type(df["timestamp"][0]))
        
        #run inference
        results = []
        for out, item in zip(pipe(KeyDataset(dataset, "text"), batch_size=32, truncation = True, padding = True), dataset):
            results.append({
                "repo": item.get("repo"),
                "seq": item.get("seq"),
                "text": item.get("text"),
                "timestamp": item.get("timestamp").isoformat(),
                "cid": item.get("cid"),
                "uri": item.get("uri"),
                "label": out["label"],
                "score": out["score"]
            })
            
        #Saving the file    
        with open(f"{OUTPUT_FOLDER}/{out_filename}", "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved {out_filename}..")
        elapsed_time = datetime.datetime.now() - processing_start_time
        print(f"Processed in: {elapsed_time}\n\n")
        
        

