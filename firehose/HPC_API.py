from atproto import CAR, models, FirehoseSubscribeReposClient, parse_subscribe_repos_message
import json
import threading
import os
from datetime import datetime, timedelta
import time

#connect to client, log time of connection
client = FirehoseSubscribeReposClient()
start_time = datetime.now()
time_limit = timedelta(days=2, hours=23, minutes = 45)

# Limit messages for testing
SAVE_THRESHOLD = 100000  # Number of entries before saving
message_count = 0  # Counter for all messages
keep_running = True

# Lists to store CREATE commits
Posts = []
Likes = []
Follows = []
Reposts = []

# Lists to store Identity and Account messages
Identity = []
Account = []

def stop_firehose():
    """Stops the Firehose listener"""
    print("Stopping client..")
    client.stop()
    finishing_touches()

def finishing_touches():
    save_list_to_disk("Posts", Posts)
    save_list_to_disk("Likes", Likes)
    save_list_to_disk("Reposts", Reposts)
    save_list_to_disk("Follows", Follows)
    save_list_to_disk("Identity", Identity)
    save_list_to_disk("Account", Account)
    print(f"Total Messages Processed: {message_count}")
    
def runtime_check():
    global keep_running
    hour = 0
    while keep_running:
        elapsed_time = datetime.now() - start_time
        if elapsed_time.total_seconds() >= hour * 3600:
            print(f"{elapsed_time}: {message_count} messages processed ")
            hour += 3
        if elapsed_time >= time_limit:
            keep_running = False
            stop_firehose()
            time.sleep(3)
        time.sleep(300)

def extract_subject_info(subject):
    """Safely extracts subject CID and URI from a Main object or dictionary."""
    if hasattr(subject, "cid") and hasattr(subject, "uri"):  # Check if it's a Main object
        return str(subject.cid), subject.uri
    elif isinstance(subject, dict):  # If it's already a dictionary
        return subject.get("cid"), subject.get("uri")
    return None, None  # Fallback if structure is unexpected

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


def save_list_to_disk(list_name, data_list):
    """Saves a specific list to disk and clears the saved portion."""
    directory = f"../data/backups/{list_name.lower()}"
    os.makedirs(directory, exist_ok=True)
    counter = get_next_counter(directory)
    filename = f"{directory}/{list_name.lower()}_backup_{counter}.json"
    mode = "a" if os.path.exists(filename) else "w"  # Append if file exists
    with open(filename, mode, encoding="utf-8") as file:
        json.dump(data_list[:SAVE_THRESHOLD], file, ensure_ascii=False)
        file.write("\n")  # Newline separator for appending

    # Clear saved portion from memory
    del data_list[:SAVE_THRESHOLD]


def on_message_handler(message) -> None:
    """Processes Firehose messages and categorizes them."""
    global message_count
    global keep_running

    processed = False  # Track if at least one valid message is processed

    try:
        commit = parse_subscribe_repos_message(message)

        if isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            if not commit.blocks:
                return

            car = CAR.from_bytes(commit.blocks)

            for op in commit.ops:
                if op.action != "create" or not op.cid:
                    continue

                record_raw_data = car.blocks.get(op.cid)
                if not record_raw_data:
                    continue

                record = models.get_or_create(record_raw_data, strict=False)
                if not record:
                    continue

                # Extract event type from path
                event_type = op.path.split("/")[0]

                # Convert record to dictionary safely
                record_data = record.__dict__

                # Process event types
                if event_type == "app.bsky.feed.post":
                    Posts.append({
                        "repo": commit.repo,
                        "timestamp": commit.time,
                        "seq": commit.seq,
                        "text": record_data.get("text"),
                        "langs": record_data.get("langs", []),
                        "cid": str(op.cid),
                        "uri": f"at://{commit.repo}/{op.path}",
                    })
                    processed = True
                if event_type == "app.bsky.feed.like":
                    liked_post_cid, liked_post_uri = extract_subject_info(record_data.get("subject"))
                    Likes.append({
                        "repo": commit.repo,
                        "timestamp": commit.time,
                        "seq": commit.seq,
                        "liked_post_cid": liked_post_cid,
                        "liked_post_uri": liked_post_uri,
                    })
                    processed = True
                if event_type == "app.bsky.feed.repost":
                    reposted_post_cid, reposted_post_uri = extract_subject_info(record_data.get("subject"))
                    Reposts.append({
                        "repo": commit.repo,
                        "timestamp": commit.time,
                        "seq": commit.seq,
                        "reposted_post_cid": reposted_post_cid,
                        "reposted_post_uri": reposted_post_uri,
                    })
                    processed = True
                if event_type == "app.bsky.graph.follow":
                    followed_user = record_data.get("subject") if isinstance(record_data.get("subject"), str) else None
                    Follows.append({
                        "repo": commit.repo,
                        "timestamp": commit.time,
                        "seq": commit.seq,
                        "followed_user": followed_user,
                    })
                    processed = True

            if processed:
                message_count += 1  # Only increment if at least one valid operation was processed

                # Save only the lists that have reached the threshold
                if len(Posts) >= SAVE_THRESHOLD:   
                    save_list_to_disk("Posts", Posts)
                if len(Likes) >= SAVE_THRESHOLD:
                    save_list_to_disk("Likes", Likes)
                if len(Reposts) >= SAVE_THRESHOLD:
                    save_list_to_disk("Reposts", Reposts)
                if len(Follows) >= SAVE_THRESHOLD:
                    save_list_to_disk("Follows", Follows)

        elif hasattr(message, "header") and hasattr(message.header, "t"):
            # Handle identity and account messages
            event_type = message.header.t.lower()

            if event_type == "#identity":
                Identity.append(getattr(message, "body", {}))  # Extract the actual dict content
                processed = True
            elif event_type == "#account":
                Account.append(getattr(message, "body", {}))  # Extract the actual dict content
                processed = True

            if processed:
                message_count += 1  # Increment message count for identity/account messages

                # Save Identity and Account lists only if they reach the threshold
                if len(Identity) >= SAVE_THRESHOLD:
                    save_list_to_disk("Identity", Identity)
                if len(Account) >= SAVE_THRESHOLD:
                    save_list_to_disk("Account", Account)
            

    except Exception as e:
        print(f"Error processing message: {e}")
        
def start_firehose():
    client.start(on_message_handler)
    
# Start Firehose stream
firehose_thread = threading.Thread(target=start_firehose)
time_check_thread = threading.Thread(target=runtime_check)
firehose_thread.start()
time_check_thread.start()

    

