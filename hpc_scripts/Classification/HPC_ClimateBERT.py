from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import load_dataset
import json
import os
import datetime
import time
import torch

FILTERED_FOLDER = "../../data/filtered"
OUTPUT_FOLDER = "../../data/climate_classified"
KEEP_RUNNING = True
MAX_RUNTIME = datetime.timedelta(days = 2, hours = 23, minutes = 25)
BATCH_SIZE = 32

dataset_name = "climatebert/climate_detection"
model_name = "climatebert/distilroberta-base-climate-detector"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

os.makedirs(OUTPUT_FOLDER, exist_ok = True)
start_time = datetime.datetime.now()

#check CUDA stuff of HPC
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.version.cuda) 

while KEEP_RUNNING:
    found_new_file = False
    files = os.listdir(FILTERED_FOLDER)
    for filename in files:
        number = filename.split("_")[-1].replace(".json", "")
        out_filename = f"climate_classified_posts_{number}.json"
        
        #Check if already processed
        if os.path.exists(f"{OUTPUT_FOLDER}/{out_filename}"):
            print(f"Skipping {filename} (already processed)")
            continue
        
        found_new_file = True
        processing_start_time = datetime.datetime.now()
        print(f"processing: {filename}\n")
        file_path = f"{FILTERED_FOLDER}/{filename}"
        dataset = load_dataset("json", data_files=file_path, split="train")
        
        #run inference
        results = []
        for out, item in zip(pipe(KeyDataset(dataset, "text"), batch_size=BATCH_SIZE, truncation = True, padding = True), dataset):
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
        
    overall_time = datetime.datetime.now() - start_time
    if overall_time > MAX_RUNTIME:
        KEEP_RUNNING = False
        print(f"Max runtime exceeded ({overall_time}), stopping the process.")
    if not found_new_file:
        print("No new file found, sleeping for 30 mins")
        time.sleep(1800)
        
print("\nfinished")

