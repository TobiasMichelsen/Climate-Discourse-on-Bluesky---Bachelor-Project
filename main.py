from database.database import setup_database
from firehose.listener import start_firehose

if __name__ == "__main__":
    print("🚀 Initializing Firehose Scraper...")

    # ✅ Ensure database tables are created before starting the listener
    setup_database()

    # ✅ Start the Firehose Listener
    print("📡 Starting Firehose Stream...")
    start_firehose()
