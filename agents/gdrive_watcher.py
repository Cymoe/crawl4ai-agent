import os
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from .crawl_gdrive_docs import process_file

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1PaIeRdrUefDcyrXSYQ-CJlNATIbDZhst'
PROCESSED_FILES_PATH = 'processed_files.json'
CHECK_INTERVAL = 300  # 5 minutes in seconds

class DriveWatcher:
    def __init__(self):
        self.processed_files: Dict[str, str] = {}  # file_id -> last_modified
        self.load_processed_files()
        
    def load_processed_files(self):
        """Load the list of processed files from disk."""
        try:
            if os.path.exists(PROCESSED_FILES_PATH):
                with open(PROCESSED_FILES_PATH, 'r') as f:
                    self.processed_files = json.load(f)
        except Exception as e:
            print(f"Error loading processed files: {e}")
            self.processed_files = {}
    
    def save_processed_files(self):
        """Save the list of processed files to disk."""
        try:
            with open(PROCESSED_FILES_PATH, 'w') as f:
                json.dump(self.processed_files, f)
        except Exception as e:
            print(f"Error saving processed files: {e}")
    
    def get_credentials(self):
        """Initialize and return Google Drive credentials using service account."""
        try:
            # Get the service account key JSON from environment variable
            service_account_info = json.loads(os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY', '{}'))
            if not service_account_info:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set or invalid")
            
            creds = Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES
            )
            return creds
        except Exception as e:
            print(f"Error getting credentials: {e}")
            return None
    
    async def check_for_changes(self):
        """Check for new or modified files in the Drive folder."""
        try:
            service = build('drive', 'v3', credentials=self.get_credentials())
            
            # List all files in the folder
            results = service.files().list(
                q=f"'{FOLDER_ID}' in parents",
                fields="files(id, name, mimeType, modifiedTime)"
            ).execute()
            
            current_files = results.get('files', [])
            
            for file in current_files:
                file_id = file['id']
                modified_time = file['modifiedTime']
                
                # Check if file is new or modified
                if (file_id not in self.processed_files or 
                    self.processed_files[file_id] != modified_time):
                    print(f"\nProcessing file: {file['name']}")
                    await process_file(service, file)
                    self.processed_files[file_id] = modified_time
                    self.save_processed_files()
            
            # Check for deleted files
            stored_ids = set(self.processed_files.keys())
            current_ids = {f['id'] for f in current_files}
            deleted_ids = stored_ids - current_ids
            
            if deleted_ids:
                print(f"\nDetected {len(deleted_ids)} deleted files")
                for file_id in deleted_ids:
                    del self.processed_files[file_id]
                self.save_processed_files()
                
        except Exception as e:
            print(f"Error checking for changes: {e}")

async def main():
    watcher = DriveWatcher()
    print(f"Starting Drive Watcher (checking every {CHECK_INTERVAL} seconds)")
    
    while True:
        print(f"\n=== Checking for changes at {datetime.now()} ===")
        await watcher.check_for_changes()
        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    asyncio.run(main())
