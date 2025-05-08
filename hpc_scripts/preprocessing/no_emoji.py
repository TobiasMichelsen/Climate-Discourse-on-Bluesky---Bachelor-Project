import json
import os
import regex as re
import time

backups_folder = "backups/posts"
processed_folder = "processed"
os.makedirs(processed_folder, exist_ok=True)

# Only process files older than this many seconds (e.g. 120s = 2 minutes)
MIN_FILE_AGE_SECONDS = 30*60 #30 mins

def clean_case_1(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    cleaned = re.sub(r"[^\p{Latin}0-9\s.,!?%°℃'’\"-]", '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

english_langs = ["en", "ca", "en-US", "uk", "en-AU", "en-GB", "en-UK", "en-CA", "en-us"]

for filename in sorted(os.listdir(backups_folder)):
    if filename.startswith("posts_backup_") and filename.endswith(".json"):
        number = filename.split("_")[-1].replace(".json", "")
        output_filename = f"posts_no_emojis_{number}.json"
        output_path = os.path.join(processed_folder, output_filename)
        input_path = os.path.join(backups_folder, filename)

        # Check if already processed
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already processed)")
            continue

        # Check if file is too new
        file_age = time.time() - os.path.getmtime(input_path)
        if file_age < MIN_FILE_AGE_SECONDS:
            print(f"Skipping {filename} (too recent, age: {int(file_age)}s)")
            continue

        # Safe to process
        with open(input_path, "r", encoding="utf-8") as f:
            all_posts = json.load(f)

        english_posts = []

        for post in all_posts:
            langs = post.get("langs")
            text = (post.get("text") or "").strip()

            if langs and all(lang in english_langs for lang in langs):
                cleaned_text = clean_case_1(text)
                if cleaned_text:
                    post["text"] = cleaned_text
                    english_posts.append(post)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(english_posts, f, indent=2, ensure_ascii=False)

        print(f"Processed {filename} → {output_filename} ({len(english_posts)} posts)")
