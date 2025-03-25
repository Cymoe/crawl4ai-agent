import os
import json
import base64
import pickle
import asyncio
import traceback
from datetime import datetime
import logging
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

from .crawl_gdrive_docs import process_file, delete_file_from_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DriveWatcher")

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
CHECK_INTERVAL = 30  # seconds

class DriveWatcher:
    """Watches a Google Drive folder for changes and processes new/modified files."""
    
    def __init__(self):
        """Initialize the DriveWatcher."""
        self.processed_files = {}
        self.load_processed_files()
        logger.info(f"DriveWatcher initialized with folder ID: {FOLDER_ID}")
        logger.info(f"Loaded {len(self.processed_files)} previously processed files")
    
    def load_processed_files(self):
        """Load the list of processed files from disk."""
        try:
            if os.path.exists('processed_files.json'):
                with open('processed_files.json', 'r') as f:
                    self.processed_files = json.load(f)
                logger.info(f"Loaded {len(self.processed_files)} processed files from disk")
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
    
    def save_processed_files(self):
        """Save the list of processed files to disk."""
        try:
            with open('processed_files.json', 'w') as f:
                json.dump(self.processed_files, f)
            logger.info(f"Saved {len(self.processed_files)} processed files to disk")
        except Exception as e:
            logger.error(f"Error saving processed files: {e}")
    
    def get_credentials(self):
        """Get Google Drive API credentials."""
        try:
            service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')
            if not service_account_key:
                logger.error("Error: GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set")
                return None
            
            # Decode the base64-encoded service account key
            try:
                service_account_info = json.loads(base64.b64decode(service_account_key))
            except Exception as e:
                logger.error(f"Error decoding service account key: {e}")
                return None
            
            # Create credentials from the service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES)
            
            return credentials
        except Exception as e:
            logger.error(f"Error getting credentials: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def check_for_changes(self):
        """Check for new or modified files in the Drive folder."""
        try:
            if not FOLDER_ID:
                logger.error("Error: GOOGLE_DRIVE_FOLDER_ID environment variable not set")
                return
                
            credentials = self.get_credentials()
            if not credentials:
                logger.error("Failed to get credentials")
                return
                
            service = build('drive', 'v3', credentials=credentials)
            
            # List all files in the folder
            try:
                results = service.files().list(
                    q=f"'{FOLDER_ID}' in parents",
                    fields="files(id, name, mimeType, modifiedTime)"
                ).execute()
            except HttpError as e:
                logger.error(f"HTTP error listing files: {e}")
                logger.error(traceback.format_exc())
                return
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                logger.error(traceback.format_exc())
                return
            
            current_files = results.get('files', [])
            logger.info(f"=== Found {len(current_files)} files in Drive folder ===")
            
            # Track files we've seen in this run
            seen_files = set()
            
            for file in current_files:
                file_id = file['id']
                file_name = file['name']
                modified_time = file['modifiedTime']
                mime_type = file.get('mimeType', 'unknown')
                
                seen_files.add(file_id)
                
                # Log file details
                logger.info(f"File: {file_name} (ID: {file_id})")
                logger.info(f"  MIME Type: {mime_type}")
                logger.info(f"  Modified: {modified_time}")
                
                # Check if file is new or modified
                if (file_id not in self.processed_files or 
                    self.processed_files[file_id] != modified_time):
                    logger.info(f">>> Processing new/modified file: {file_name} (ID: {file_id})")
                    try:
                        await process_file(service, file)
                        self.processed_files[file_id] = modified_time
                        self.save_processed_files()
                        logger.info(f"Successfully processed file: {file_name}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_name}: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.info(f"Skipping unchanged file: {file_name}")
            
            # Check for deleted files
            deleted_files = []
            for file_id in self.processed_files:
                if file_id not in seen_files:
                    try:
                        # Try to get the file to confirm it's deleted
                        try:
                            service.files().get(fileId=file_id).execute()
                            # If we get here, file still exists but not in our folder
                            logger.info(f"File {file_id} exists but not in target folder")
                        except HttpError as e:
                            if e.resp.status == 404:
                                # File is truly deleted
                                logger.info(f"File {file_id} has been deleted from Drive")
                                deleted_files.append(file_id)
                                # Delete from database
                                await delete_file_from_database(file_id)
                    except Exception as e:
                        logger.error(f"Error checking deleted file {file_id}: {e}")
                        logger.error(traceback.format_exc())
            
            # Remove deleted files from our tracking
            for file_id in deleted_files:
                if file_id in self.processed_files:
                    del self.processed_files[file_id]
            
            if deleted_files:
                self.save_processed_files()
                logger.info(f"Removed {len(deleted_files)} deleted files from tracking")
            
            logger.info("=== Finished checking for changes ===")
            
            # Wait before checking again
            await asyncio.sleep(CHECK_INTERVAL)
            
            # Recursively check again
            await self.check_for_changes()
            
        except Exception as e:
            logger.error(f"Error in check_for_changes: {e}")
            logger.error(traceback.format_exc())

async def main():
    watcher = DriveWatcher()
    await watcher.check_for_changes()

if __name__ == '__main__':
    asyncio.run(main())
