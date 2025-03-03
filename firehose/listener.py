from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, models
from database.database import get_connection  # Import database functions
from firehose.interested_records import get_interested_records  # Import interested records
import json

MAX_MESSAGES = 50  # Limit for testing
message_count = 0
client = FirehoseSubscribeReposClient()

# Load interested records dynamically
INTERESTED_RECORDS = get_interested_records()

def on_message_handler(message) -> None:
    global message_count
    if MAX_MESSAGES and message_count >= MAX_MESSAGES:
        print("Reached message limit. Stopping Firehose.")
        client.stop()
        return

    conn = get_connection()
    cur = conn.cursor()

    try:
        commit = parse_subscribe_repos_message(message)
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return

        if not commit.blocks:
            return

        car = CAR.from_bytes(commit.blocks)

        for op in commit.ops:
            if op.action != 'create' or not op.cid:
                continue

            uri = f"at://{commit.repo}/{op.path}"
            record_raw_data = car.blocks.get(op.cid)
            if not record_raw_data:
                continue

            record = models.get_or_create(record_raw_data, strict=False)

            # Extract event type from the URI path
            event_type = op.path.split("/")[0]

            # DEBUGGING: Print event type to see what's happening
            print(f"🔍 Processing event type: {event_type}")

            # Check if event type exists in INTERESTED_RECORDS
            record_type = INTERESTED_RECORDS.get(event_type)

            if record_type is None:
                print(f"⚠️ Unknown event type: {event_type} (Skipping)")
                continue

            # **Fix: Ensure record_type is a valid type before calling isinstance**
            if not isinstance(record_type, type):
                print(f"❌ ERROR: record_type for {event_type} is not a valid type: {record_type}")
                continue

            # Insert into the database dynamically
            cur.execute(f"""
                INSERT INTO {event_type.replace('.', '_')} (timestamp, repo, data)
                VALUES (?, ?, ?)
            """, (commit.time, commit.repo, json.dumps(record_raw_data)))

            conn.commit()

            print(f"✅ Stored {event_type} from {commit.repo}")

            message_count += 1

    except Exception as e:
        print(f"❌ Error processing message: {e}")

    finally:
        conn.close()


def start_firehose():
    """Starts the Firehose listener when called from main.py."""
    print("📡 Firehose listener started...")
    client.start(on_message_handler)
