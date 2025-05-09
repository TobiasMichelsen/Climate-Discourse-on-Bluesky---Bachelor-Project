from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import pandas as pd
import pickle
import datetime
from huggingface_hub import login
import torch
import sys

#variables
input_path = "../BERTopic/logs"
filename = "all-MiniLM-L6-v2_cosine_c100_s5_nt18_u5-5-0.0/topics_before_reduction.json"
keep_running = True
counter = 0
concat_df = pd.DataFrame()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("data/conf", exist_ok=True)  

#check torch verison
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

#CUDA check
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device count:", torch.cuda.device_count(), flush=True)
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA", flush=True)




#Initialize model
HUGGINGFACE_TOKEN = "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"
print(f"Initializing model..", flush=True)
login(token=HUGGINGFACE_TOKEN)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Or LLaMA, Zephyr, etc.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)




df = pd.read_json(f"{input_path}/{filename}", lines=True)
start_time = datetime.datetime.now()
for i in df.topic.unique(): ###read text
    cluster_start_time = datetime.datetime.now()
    counter += 1
    
    df_cluster = df[df.topic == i]
    print(f"shape: {df_cluster.shape}", flush=True)
    sampled_df = df_cluster.sample(n=100, random_state=42)
                        
    text_counter = 0
    for j in range(sampled_df.shape[0]):
        cur_row = sampled_df.iloc[j]
        text = cur_row.text
        prompt_start = datetime.datetime.now()
        prompt = f"""
        You are a helpful assistant classifying climate-related texts into one or more relevant categories from a fixed list. 
        categories with example sentences:
        [
        - Activism: "Thousands marched in the climate strike demanding government action."
        - Agriculture: "Crops are failing due to prolonged droughts intensified by climate change."
        - Diet: "Plant-based diets are gaining popularity for their environmental benefits."
        - Disaster: "Wildfires have devastated thousands of acres in California."
        - Fossil: "Oil prices continue to rise amid geopolitical tensions and supply constraints."
        - Health: "Air pollution is linked to rising asthma rates in urban areas."
        - Hydropower: "Drought conditions are affecting the electricity output of hydropower plants in Brazil."
        - Justice: "Communities of color are disproportionately affected by environmental hazards."
        - Lifestyle: "People are embracing minimalist living to reduce their carbon footprint."
        - Nature: "Pollinators like bees are disappearing, threatening global food security."
        - Politics: "The new administration reversed several environmental protections."
        - Science: "Scientists warn that ice sheets are melting faster than expected."
        - Solar: "Government subsidies are making rooftop solar panels more accessible."
        - Waste: "Cities are expanding composting and recycling programs to reduce landfill use."
        - Weather: "Global temperatures hit a new record high this year."
        - Wind: "Community wind projects are helping rural areas become energy independent."
        ]

        Instructions:
        - Carefully analyze the text.
        - Select all the categories, strictly from the above list that apply.
        - For each, assign a confidence score based on how clearly the topic is discussed.
        - Output only the categories with confidence scores 
        - Output should be in this format:
        {{"Fossil Fuels": 0.9,"Politics": 0.8}}
        
        Classify this text:
        {text}
        response:
        """
        try:
            response = pipe(prompt, max_new_tokens=50)[0]["generated_text"]
            response = response.split("response:")[-1].strip()
        except Exception as e:
            print(f"error prompting Minstral:\ntext:\n{text}\nerror:\n{e}", flush=True)
            continue
        out_df = pd.DataFrame({"cluster":[i],"text": [text], "response": [response]})
        concat_df = pd.concat([concat_df, out_df])
        if text_counter % 10 == 0:
            with open("data/conf/LLM_classified_df.pkl", "wb") as f:
                pickle.dump(concat_df, f)
        text_counter += 1
        prompt_time = datetime.datetime.now() - prompt_start
        print(f"processed {text_counter} text in cluster {i} dur.: {prompt_time}", flush=True)
    elapsed_time = datetime.datetime.now() - cluster_start_time
    print(f"finished processing cluster {i} \nit took: {elapsed_time}", flush=True)


total_time = datetime.datetime.now() - start_time
print(f"processed {counter} clusters in {total_time}", flush=True)
print("\nFinished!", flush=True)