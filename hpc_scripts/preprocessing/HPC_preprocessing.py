import os
import pandas as pd
import regex as re
import json
import time
import datetime



#variables
ENGLISH_LANGS = ["en", "ca", "en-US", "uk", "en-AU", "en-GB", "en-UK", "en-CA", "en-us"]
MIN_TEXT_LEN = 60
MAX_RUNTIME = datetime.timedelta(days= 2, hours = 23, minutes = 15)
POSTS_FOLDER = "../Firehose/backups/posts"
OUTPUT_FOLDER = "../data/filtered"
ARTIFACTS_FOLDER = "artifacts"
PARTIAL_DF_PATH = os.path.join(ARTIFACTS_FOLDER, "partial_concat_df.json")
PROCESSED_FILE_NUMBERS_PATH = os.path.join(ARTIFACTS_FOLDER, "processed_file_numbers.json")


#functions
def remove_non_latin(text): #change name to remove_non_latin()
    if text:
        text = text.replace('\n', ' ').replace('\r', ' ')
        cleaned = re.sub(r"[^\p{Latin}0-9\s.,!?%°℃'’\"-]", '', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

def get_next_counter(directory):
    """Finds the next available counter for the given list and date inside a folder."""

    existing_files = os.listdir(directory)
    
    # Extract numbers from filenames
    counters = []
    for fname in existing_files:
        parts = fname.split("_")
        try:
            num = int(parts[2].split(".")[0])  # Extract counter from "listname_backup_X_MM-DD.json"
            counters.append(num)
        except ValueError:
            continue  # Skip files that don’t match pattern

    return max(counters, default=0) + 1  # Start at 1 if no files exist

def load_processed_files():
    # Check if the processed files list exists
    if os.path.exists(PROCESSED_FILE_NUMBERS_PATH):
        with open(PROCESSED_FILE_NUMBERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # If the file doesn't exist, create an empty list
        return []

def save_processed_files(processed_files):
    with open(PROCESSED_FILE_NUMBERS_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_files, f)

def clean(post_file):
    # print(f"df shape: {post_file.shape}",flush=True)

    df = post_file

    #removing non english langs
    df = df[df["langs"].apply(lambda langs: isinstance(langs, list) and len(langs) > 0)]
    # print(f"df shape after dropping empty langs: {df.shape}",flush=True)
    df = df[df["langs"].apply(lambda langs: all(lang in ENGLISH_LANGS for lang in langs))]
    # print(f"df shape after removing non english langs: {df.shape}",flush=True)

    #removing based on text length
    df = df.dropna(subset=["text"]) #drop na as Nonetype has no length
    df = df[df["text"].apply(len) >= MIN_TEXT_LEN]
    # print(f"df shape after min text len cutoff: {df.shape}",flush=True)

    df["text"] = df["text"].apply(remove_non_latin)  
    df = df.dropna(subset=["text"])
    df = df[df["text"].apply(len) >= MIN_TEXT_LEN]
    # print(f"df shape after non latin removal: {df.shape}",flush=True)

    return df


    
#CLEANING ON DF


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
    processed_files = load_processed_files()
    start_time = datetime.datetime.now()
    if os.path.exists(PARTIAL_DF_PATH):
        print("Resuming from previous partial concat_df...",flush=True)
        concat_df = pd.read_json(PARTIAL_DF_PATH)
    else:
        concat_df = pd.DataFrame()
    while True:
        elapsed_time = datetime.datetime.now() - start_time
        if elapsed_time > MAX_RUNTIME:
            print("Max job time reached. Saving partial df and exiting...",flush=True)
            concat_df.to_json(PARTIAL_DF_PATH, orient="records", lines=False, force_ascii=False, indent=2)
            print(f"Saved {concat_df.shape[0]} rows to {PARTIAL_DF_PATH}",flush=True)
            break  # Exit the job cleanly
        new_file_found = False   #Flag to track if any new file was processed this cycle
        num_files = len(os.listdir(POSTS_FOLDER))
        for i in range(1,num_files+1):
            for filename in os.listdir(POSTS_FOLDER):
                if filename.startswith("posts_backup_") and filename.endswith(".json"):
                    
                    #Check if processed already
                    number = filename.split("_")[-1].replace(".json", "")
                    if number in processed_files:
                        continue
                    #ensure reading files in order
                    if int(number) != i:
                        continue
                    
                    #Found file to process
                    new_file_found = True
                    print(f"\nProcessing: {filename}",flush=True)
                    # Safe to process
                    with open(f"{POSTS_FOLDER}/{filename}", "r", encoding="utf-8") as f:
                        post_file = pd.read_json(f)
                    cleaned_df = clean(post_file)
                    
                    #Saving processed file number
                    processed_files.append(number)
                    save_processed_files(processed_files)
                    
                    #Save in 100.000 installments
                    concat_df = pd.concat([concat_df, cleaned_df], ignore_index=True)
                    if concat_df.shape[0] >= 100000:
                        #Get counter of files in the output folder
                        counter = get_next_counter(OUTPUT_FOLDER) 
                        output_filename = f"posts_filtered_{counter}.json"
                        df_to_save = concat_df[:100000]
                        df_to_save.to_json(f"{OUTPUT_FOLDER}/{output_filename}", orient="records", lines=False, force_ascii=False, indent=2)
                        print("Dataframe saved containing 100k rows, YAY",flush=True)
                        concat_df = concat_df[100000:]
        if not new_file_found:
            print("No new files found. Sleeping for 30 mins..\n",flush=True)
            time.sleep(1800)
    print("Job finished processing",flush=True)
    
#RUN            
if __name__ == "__main__":
    main()
