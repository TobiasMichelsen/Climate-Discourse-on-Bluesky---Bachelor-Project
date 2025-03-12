from atproto import CAR, models, FirehoseSubscribeReposClient, parse_subscribe_repos_message
import json
import threading
import sqlite3
import base64
import time
import sys

client = FirehoseSubscribeReposClient()

sys.stdout.reconfigure(encoding='utf-8')
MAX_MESSAGES = 1000  # Limit messages for testing
message_count = 0  # Counter
storing_count = 0
stop_storing = False
firehose_data = []  # List to store structured messages


def stop_firehose():
    """Stops the Firehose listener after MAX_MESSAGES messages."""
    print("✅ Reached message limit. Stopping Firehose.")
    client.stop()

def on_message_handler(message) -> None:
    """Handler that parses Firehose messages and extracts key data"""
    global message_count
    if message_count >= MAX_MESSAGES:
        stop_firehose()
        return

    try:
        commit = parse_subscribe_repos_message(message)
        
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return

        if not commit.blocks:
            return

        car = CAR.from_bytes(commit.blocks)

        structured_data = {
            "repo": commit.repo,
            "timestamp": commit.time,
            "rev": commit.rev,
            "seq": commit.seq,
            "ops": []
        }

        for op in commit.ops:
            if op.action != "create" or not op.cid:
                continue

            uri = f"at://{commit.repo}/{op.path}"
            record_raw_data = car.blocks.get(op.cid)
            if not record_raw_data:
                continue

            record = models.get_or_create(record_raw_data, strict=False)
            
            structured_data["ops"].append({
                "action": op.action,
                "path": op.path,
                "cid": str(op.cid),
                "record": record.__dict__,  # Convert record to dictionary
            })

        firehose_data.append(structured_data)  # Store structured data
        
        message_count += 1

        # If MAX_MESSAGES reached, stop Firehose in a separate thread
        if message_count >= MAX_MESSAGES:
            threading.Thread(target=stop_firehose).start()

        print(f"✅ Processed Message {message_count}/{MAX_MESSAGES} at {commit.time}")

    except Exception as e:
        print(f"❌ Error processing message: {e}")

def convert_to_dict(obj):
    """ Recursively convert non-serializable objects into JSON-compatible format. """
    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, bytes):  # 🛠️ Convert bytes to a base64 string
        return base64.b64encode(obj).decode('utf-8')
    elif hasattr(obj, "__dict__"):  # Convert custom objects with attributes
        return {key: convert_to_dict(value) for key, value in vars(obj).items()}
    else:
        return obj  # Return basic types as-is


# ✅ Function to Construct Post URI from repo and cid
def construct_post_uri(repo, cid):
    """ Constructs a Bluesky post URI from repo (DID) and CID. """
    return f"at://{repo}/app.bsky.feed.post/{cid}" if repo and cid else None

def store_data():
    global storing_count, stop_storing
    # Connect to SQLite
    conn = sqlite3.connect("firehose.db")
    cur = conn.cursor()

    # 🔥 Drop and Recreate Tables with More Structured Columns
    cur.execute("DROP TABLE IF EXISTS posts")
    cur.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            timestamp TEXT,
            seq INTEGER,
            text TEXT,
            langs TEXT,
            parent_cid TEXT,
            parent_uri TEXT,
            post_uri TEXT,  --!!! might not work !!!!
            cid TEXT,
            root_cid TEXT,
            root_uri TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS likes")
    cur.execute("""
        CREATE TABLE likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            timestamp TEXT,
            seq INTEGER,
            liked_post_cid TEXT,
            liked_post_uri TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS reposts")
    cur.execute("""
        CREATE TABLE reposts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            timestamp TEXT,
            seq INTEGER,
            reposted_post_cid TEXT,
            reposted_post_uri TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS follows")
    cur.execute("""
        CREATE TABLE follows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            timestamp TEXT,
            seq INTEGER,
            followed_user TEXT
        )
    """)

    conn.commit()
    conn.close()
    while not stop_storing:
        if firehose_data:
            message = firehose_data.pop(0)
            try:
                conn = sqlite3.connect("firehose.db")  
                cur = conn.cursor()  
                for op in message["ops"]:
                    event_type = op["path"].split("/")[0]  # Extract event type

                    # 🛠️ Ensure `record` is a dictionary (convert JSON string if needed)
                    record_data = json.loads(op["record"]) if isinstance(op["record"], str) else op["record"]

                    # ✅ Convert objects inside `record_data` into JSON-compatible format
                    record_data_serializable = convert_to_dict(record_data)

                    # Extract common fields
                    repo = message["repo"]
                    timestamp = message["timestamp"]
                    seq = message["seq"]
                    cid = op.get("cid", None)  # ✅ Extract CID from Firehose event
                    post_uri = construct_post_uri(repo, cid)  # ✅ Construct Post URI

                    if event_type == "app.bsky.feed.post":
                        text = record_data_serializable.get("text", None)
                        langs = ",".join(record_data_serializable.get("langs", [])) if record_data_serializable.get("langs") else None

                        # 🛠️ Fix: Ensure `reply` data is safely extracted
                        reply_data = record_data_serializable.get("reply", {}) or {}

                        parent_cid = reply_data.get("parent", {}).get("cid", None)
                        parent_uri = reply_data.get("parent", {}).get("uri", None)
                        root_cid = reply_data.get("root", {}).get("cid", None)
                        root_uri = reply_data.get("root", {}).get("uri", None)

                        # Insert structured data into `posts`
                        cur.execute("""
                            INSERT INTO posts (repo, timestamp, seq, text, langs, parent_cid, parent_uri, post_uri, cid, root_cid, root_uri)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (repo, timestamp, seq, text, langs, parent_cid, parent_uri, post_uri, cid, root_cid, root_uri))

                    elif event_type == "app.bsky.feed.like":
                        subject = record_data_serializable.get("subject", {})
                        liked_post_cid = subject.get("cid", None) if isinstance(subject, dict) else None
                        liked_post_uri = subject.get("uri", None) if isinstance(subject, dict) else None

                        # Insert structured data into `likes`
                        cur.execute("""
                            INSERT INTO likes (repo, timestamp, seq, liked_post_cid, liked_post_uri)
                            VALUES (?, ?, ?, ?, ?)
                        """, (repo, timestamp, seq, liked_post_cid, liked_post_uri))

                    elif event_type == "app.bsky.feed.repost":
                        subject = record_data_serializable.get("subject", {})
                        reposted_post_cid = subject.get("cid", None) if isinstance(subject, dict) else None
                        reposted_post_uri = subject.get("uri", None) if isinstance(subject, dict) else None

                        # Insert structured data into `reposts`
                        cur.execute("""
                            INSERT INTO reposts (repo, timestamp, seq, reposted_post_cid, reposted_post_uri)
                            VALUES (?, ?, ?, ?, ?)
                        """, (repo, timestamp, seq, reposted_post_cid, reposted_post_uri))

                    elif event_type == "app.bsky.graph.follow":
                        followed_user = record_data_serializable.get("subject", None)  # Followed user is a direct string

                        # Insert structured data into `follows`
                        cur.execute("""
                            INSERT INTO follows (repo, timestamp, seq, followed_user)
                            VALUES (?, ?, ?, ?)
                        """, (repo, timestamp, seq, followed_user))
                conn.commit()
                conn.close()
                storing_count += 1
                print(f"data point {storing_count} successfully loaded")
                if storing_count >= MAX_MESSAGES:
                    stop_storing = True
            except Exception as e:
                print(f"Error storing data: {e}")
        else:
            time.sleep(1)


# Start the Firehose stream
def start_fetching_data():
    client.start(on_message_handler)
def start_storing_data():
    store_data()

# Create and start threads
fetching_thread = threading.Thread(target=start_fetching_data)
storing_thread = threading.Thread(target=start_storing_data)

fetching_thread.start()
storing_thread.start()
