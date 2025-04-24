from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import pandas as pd
import pickle
import datetime
from huggingface_hub import login

input_path = "../..//data/climate_classified"
keep_running = True
counter = 0
concat_df = pd.DataFrame()

#Initialize mdoel
HUGGINGFACE_TOKEN = "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"
print(f"Initializing model..", flush=True)
login(token=HUGGINGFACE_TOKEN)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Or LLaMA, Zephyr, etc.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


while keep_running:
    for filename in os.listdir(input_path):
        start_time = datetime.datetime.now()
        print(f"processing {filename}..\nstart at: {start_time}", flush=True)
        counter += 1
        if counter > 3:
            keep_running = False
            break
        df = pd.read_json(f"{input_path}/{filename}")    
        df = df[df["label"] == "yes"]
        df = df[df["score"] >= .99]
        df_text = df["text"]
        for text in df_text:
            prompt = f"""
            You are a helpful assistant that classifies text into climate-related categories.
            Categories:
            1. Renewable Energy - e.g., "Solar panels are being installed across rooftops"
            2. Fossil Fuels - e.g., "Oil prices continue to rise"
            3. Agriculture - e.g., "Crops are failing due to drought"

            Classify the following:
            {text}

            Answer:
            """

            response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
            print(f"input:{text}\noutput:{response}\n", flush = True)
            out_df = pd.DataFrame({"text": [text], "response": [response]})
            concat_df = pd.concat([concat_df, out_df])
        elapsed_time = datetime.datetime.now() - start_time
        print(f"finished processing {filename}, processed {len(df)} rows\nit took: {elapsed_time}", flush=True)

print(f"processed {counter} files", flush=True) 
           
with open("LLM_classified_df_test.pkl", "wb") as f:
    pickle.dump(concat_df, f)
print("saved output..\n finished!")