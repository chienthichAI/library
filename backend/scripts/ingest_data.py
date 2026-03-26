import os
import sys
import argparse
import requests
from pathlib import Path

# Add project root to path so we can import app if needed directly
# But for a CLI tool, it's better to use the running API
API_URL = "http://localhost:8000/api/chatbot/upload-docs"

def ingest_directory(directory_path):
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        print(f"❌ Error: Directory {directory_path} not found.")
        return

    files = list(path.glob("*.pdf")) + list(path.glob("*.csv")) + list(path.glob("*.txt"))
    if not files:
        print(f"ℹ️ No supported files (.pdf, .csv, .txt) found in {directory_path}")
        return

    print(f"🚀 Found {len(files)} documents. Starting batch ingestion...")
    
    success_count = 0
    for file_path in files:
        print(f"📤 Uploading: {file_path.name}...")
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={"file": (file_path.name, f, "application/octet-stream")}
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data.get('message')} ({data.get('chunks_created')} chunks)")
                success_count += 1
            else:
                print(f"❌ Failed: Server returned {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error uploading {file_path.name}: {e}")

    print(f"\n✨ Done! Successfully ingested {success_count}/{len(files)} documents.")
    print("🤖 Now you can ask the AI about these documents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Ingest Documents into SmartLib AI Chatbot")
    parser.add_argument("dir", help="Directory contains PDFs, CSVs or TXTs")
    
    args = parser.parse_args()
    
    # Check if backend is running
    try:
        requests.get("http://localhost:8000/health", timeout=2)
    except:
        print("❌ Error: Backend server is not running at localhost:8000.")
        print("💡 Please start the backend first: python -m app.main")
        sys.exit(1)

    ingest_directory(args.dir)
