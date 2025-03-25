import os
import time
import json
import asyncio
from datetime import datetime
from typing import Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64

from .crawl_gdrive_docs import process_file, delete_file_from_database

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
CHECK_INTERVAL = 30  # seconds

class DriveWatcher:
    def __init__(self):
        self.processed_files: Dict[str, str] = {}  # file_id -> last_modified
        self.load_processed_files()
        print("\n=== DriveWatcher Started ===")
        print(f"Monitoring folder: {FOLDER_ID}")
        print(f"Check interval: {CHECK_INTERVAL} seconds")
    
    def load_processed_files(self):
        """Load the list of processed files from disk."""
        try:
            if os.path.exists('processed_files.json'):
                with open('processed_files.json', 'r') as f:
                    self.processed_files = json.load(f)
                print(f"Loaded {len(self.processed_files)} processed files from disk")
            else:
                print("No processed files found, starting fresh")
        except Exception as e:
            print(f"Error loading processed files: {e}")
    
    def save_processed_files(self):
        """Save the list of processed files to disk."""
        try:
            with open('processed_files.json', 'w') as f:
                json.dump(self.processed_files, f)
        except Exception as e:
            print(f"Error saving processed files: {e}")
    
    def get_credentials(self):
        """Initialize and return Google Drive credentials using service account."""
        try:
            # Get the service account key from environment variable
            service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')
            if not service_account_key:
                print("Error: GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set")
                return None
            
            # Decode the base64-encoded service account key
            try:
                service_account_info = json.loads(base64.b64decode(service_account_key))
            except Exception as e:
                print(f"Error decoding service account key: {e}")
                return None
            
            # Create credentials from the service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES)
            
            return credentials
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
            print(f"\n=== Found {len(current_files)} files in Drive folder ===")
            
            for file in current_files:
                file_id = file['id']
                modified_time = file['modifiedTime']
                
                # Check if file is new or modified
                if (file_id not in self.processed_files or 
                    self.processed_files[file_id] != modified_time):
                    print(f"\n>>> Processing new/modified file: {file['name']} (ID: {file_id})")
                    print(f"Last modified: {modified_time}")
                    await process_file(service, file)
                    self.processed_files[file_id] = modified_time
                    self.save_processed_files()
                else:
                    print(f"Skipping unchanged file: {file['name']}")
            
            # Check for deleted files
            stored_ids = set(self.processed_files.keys())
            current_ids = {f['id'] for f in current_files}
            deleted_ids = stored_ids - current_ids
            
            if deleted_ids:
                print(f"\nDetected {len(deleted_ids)} deleted files")
                for file_id in deleted_ids:
                    # Delete from database
                    await delete_file_from_database(file_id)
                    # Remove from processed files
                    del self.processed_files[file_id]
                self.save_processed_files()
            
            # Wait before checking again
            await asyncio.sleep(CHECK_INTERVAL)
            
            # Recursively check again
            await self.check_for_changes()
            
        except HttpError as error:
            print(f"HTTP error occurred: {error}")
            print(f"Stack trace: {traceback.format_exc()}")
            await asyncio.sleep(CHECK_INTERVAL)
            await self.check_for_changes()
        except Exception as e:
            print(f"Error checking for changes: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            await asyncio.sleep(CHECK_INTERVAL)
            await self.check_for_changes()

async def main():
    watcher = DriveWatcher()
    await watcher.check_for_changes()

if __name__ == '__main__':
    asyncio.run(main())
