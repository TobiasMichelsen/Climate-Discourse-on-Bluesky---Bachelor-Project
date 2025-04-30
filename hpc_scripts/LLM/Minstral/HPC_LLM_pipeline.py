from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import pandas as pd
import pickle
import datetime
from huggingface_hub import login

#variables
input_path = "../BERTopic/logs"
filename = "all-MiniLM-L6-v2_cosine_c100_s5_nt18_u5-5-0.0/topics_before_reduction.json"
keep_running = True
counter = 0
concat_df = pd.DataFrame()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("data", exist_ok=True)  


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
        seq = cur_row.seq
        prompt_start = datetime.datetime.now()
        prompt = f"""
        You are a helpful assistant that classifies text into climate-related categories.
        categories:
        1. Renewable Energy - "Solar panels are being installed across rooftops"
        2. Fossil Fuels - "Oil prices continue to rise"
        3. Agriculture -  "Crops are failing due to drought"
        4. Lifestyle - "People are embracing minimalist living to reduce their carbon footprint."
        5. Diet - "Plant-based diets are gaining popularity for their environmental benefits."
        6. Nature - "Bird populations are declining due to habitat loss in forests."
        7. Activism - "Thousands marched in the climate strike demanding government action."
        8. Climate - "Global temperatures hit a new record high this year."
        9. Disaster - "Wildfires have devastated thousands of acres in California."
        10. Health - "Air pollution is linked to rising asthma rates in urban areas."
        11. Justice - "Communities of color are disproportionately affected by environmental hazards."
        12. Science - "Scientists warn that ice sheets are melting faster than expected."
        13. Politics - "The new administration reversed several environmental protections."
        14. Waste - "Plastic waste continues to pollute oceans despite global efforts."
        
        If you think the text belongs to a new category, not listed above, assign it to the new category. 
        Your response should only contain the corresponding categories without explanations, if you think a text belongs to multiple categories return more
        

        Classify the following:
        {text}

        response:
        """
        try:
            response = pipe(prompt, max_new_tokens=50)[0]["generated_text"]
            response = response.split("response:")[-1].strip()
        except Exception as e:
            print(f"error prompting Minstral:\ntext:\n{text}\nerror:\n{e}", flush=True)
            continue
        out_df = pd.DataFrame({"seq":[seq],"cluster":[i],"text": [text], "response": [response]})
        concat_df = pd.concat([concat_df, out_df])
        if text_counter % 10 == 0:
            with open("data/LLM_classified_df.pkl", "wb") as f:
                pickle.dump(concat_df, f)
        text_counter += 1
        prompt_time = datetime.datetime.now() - prompt_start
        print(f"processed {text_counter} text in cluster {i} dur.: {prompt_time}", flush=True)
    elapsed_time = datetime.datetime.now() - cluster_start_time
    print(f"finished processing cluster {i} \nit took: {elapsed_time}", flush=True)


total_time = datetime.datetime.now() - start_time
print(f"processed {counter} clusters in {total_time}", flush=True)
print("\nFinished!", flush=True)