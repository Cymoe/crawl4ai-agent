import os
import asyncio
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
import requests
import json
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

async def list_gdrive_files():
    """List all Google Drive files stored in the database."""
    print("\n=== Files in Database ===")
    try:
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        print(f"\nFound {len(gdrive_data)} Google Drive files:")
        for item in gdrive_data:
            print(f"\nFile: {item.get('title')}")
            print(f"Type: {item.get('metadata', {}).get('type', 'unknown')}")
            print(f"URL: {item.get('url')}")
            print(f"ID: {item.get('id')}")
            print(f"Metadata: {json.dumps(item.get('metadata', {}), indent=2)}")
            print(f"Summary: {item.get('summary')}")
            print("-" * 50)
            
        return gdrive_data
    except Exception as e:
        print(f"Error listing files: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        return []

async def check_railway_status():
    """Check if running on Railway."""
    print("\n=== Railway Deployment Status ===")
    print("Note: This is a simplified check and may not be accurate.")
    
    railway_service_url = os.getenv("RAILWAY_SERVICE_URL")
    railway_static_url = os.getenv("RAILWAY_STATIC_URL")
    
    if railway_service_url:
        print(f"Railway service URL: {railway_service_url}")
    else:
        print("No Railway service URL found. You may be running locally.")
    
    if os.getenv("RAILWAY_PROJECT_ID"):
        print(f"Running in Railway project: {os.getenv('RAILWAY_PROJECT_ID')}")
    else:
        print("Not running in Railway environment.")
    
    # Check if Supabase credentials are set
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
        print("\nSupabase credentials found. Database operations should work.")
    else:
        print("\nWARNING: Supabase credentials missing. Database operations will fail.")

async def list_actual_gdrive_files():
    """List files directly from Google Drive."""
    print("\n=== Files in Google Drive Folder ===")
    
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        print("ERROR: GOOGLE_DRIVE_FOLDER_ID not set in environment variables.")
        return []
    
    try:
        # Get credentials
        service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')
        if not service_account_key:
            print("ERROR: GOOGLE_SERVICE_ACCOUNT_KEY not set in environment variables.")
            return []
        
        # Handle both JSON and base64 encoded formats
        if service_account_key.strip().startswith('{'):
            service_account_info = json.loads(service_account_key)
        else:
            service_account_info = json.loads(base64.b64decode(service_account_key))
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # Build the Drive API client
        service = build('drive', 'v3', credentials=credentials)
        
        # List files in the folder
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("No files found in the specified Google Drive folder.")
            return []
            
        print(f"\nFound {len(files)} files in Google Drive folder {folder_id}:")
        for file in files:
            print(f"\nFile: {file.get('name')}")
            print(f"ID: {file.get('id')}")
            print(f"MIME Type: {file.get('mimeType')}")
            print(f"Modified: {file.get('modifiedTime')}")
            print("-" * 50)
            
        return files
    except Exception as e:
        print(f"Error listing Google Drive files: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")
        return []

async def compare_files():
    """Compare files in database with files in Google Drive."""
    print("\n=== Comparing Database with Google Drive ===")
    
    db_files = await list_gdrive_files()
    drive_files = await list_actual_gdrive_files()
    
    if not db_files or not drive_files:
        print("Cannot compare files: either database or Google Drive files list is empty.")
        return
    
    # Create dictionaries for easier comparison
    db_files_dict = {item.get('title'): item for item in db_files}
    drive_files_dict = {file.get('name'): file for file in drive_files}
    
    # Files in database but not in Drive
    missing_in_drive = set(db_files_dict.keys()) - set(drive_files_dict.keys())
    if missing_in_drive:
        print("\nFiles in database but not found in Google Drive:")
        for file_name in missing_in_drive:
            print(f"- {file_name}")
    
    # Files in Drive but not in database
    missing_in_db = set(drive_files_dict.keys()) - set(db_files_dict.keys())
    if missing_in_db:
        print("\nFiles in Google Drive but not in database (not yet processed):")
        for file_name in missing_in_db:
            print(f"- {file_name}")
    
    # Files in both
    common_files = set(db_files_dict.keys()) & set(drive_files_dict.keys())
    print(f"\nFiles present in both database and Google Drive: {len(common_files)}")
    for file_name in common_files:
        print(f"- {file_name}")

async def force_refresh():
    """Force a refresh of the database by scanning Google Drive."""
    print("\n=== Forcing Database Refresh ===")
    
    # Check for required environment variables
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')
    
    if not folder_id:
        print("Error: GOOGLE_DRIVE_FOLDER_ID environment variable not set")
        return
    
    if not service_account_key:
        print("Error: GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set")
        return
    
    try:
        # Decode the base64-encoded service account key
        service_account_info = json.loads(base64.b64decode(service_account_key))
        
        # Create credentials from the service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        
        # Build the Drive service
        service = build('drive', 'v3', credentials=credentials)
        
        # List all files in the folder
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType, modifiedTime)"
        ).execute()
        
        files = results.get('files', [])
        print(f"Found {len(files)} files in Google Drive folder:")
        
        for file in files:
            print(f"- {file['name']} ({file.get('mimeType', 'unknown')})")
        
        # TODO: Process files
        print("\nTo process these files, run the DriveWatcher.")
        
    except Exception as e:
        print(f"Error during refresh: {e}")
        import traceback
        print(traceback.format_exc())

async def manually_delete_file(file_name):
    """Manually delete a file from the database."""
    print(f"\n=== Manually Deleting File: {file_name} ===")
    
    from agents.crawl_gdrive_docs import delete_file_from_database
    
    try:
        await delete_file_from_database("dummy_id", file_name)
        print(f"Attempted to delete {file_name}. Check database to confirm.")
    except Exception as e:
        print(f"Error deleting file: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

async def check_supabase_functions():
    """Check if the vector search RPC functions exist in Supabase."""
    print("\n=== Checking Supabase Functions ===")
    try:
        # Check for match_documents function
        try:
            response = supabase.rpc('match_documents', {'query_embedding': []}).execute()
            print("match_documents RPC function exists!")
        except Exception as e:
            error_str = str(e)
            if "function match_documents" in error_str and "does not exist" in error_str:
                print("match_documents RPC function does NOT exist in Supabase.")
            elif "argument" in error_str and "embedding" in error_str:
                # If the error is about invalid arguments, the function exists
                print("match_documents RPC function exists!")
            else:
                print(f"Unknown error checking for match_documents: {e}")
        
        # Check for match_site_pages function
        try:
            response = supabase.rpc('match_site_pages', {'query_embedding': []}).execute()
            print("match_site_pages RPC function exists!")
            return True
        except Exception as e:
            error_str = str(e)
            if "function match_site_pages" in error_str and "does not exist" in error_str:
                print("match_site_pages RPC function does NOT exist in Supabase.")
                print("You need to create this function for vector similarity search.")
                return False
            elif "argument" in error_str and "embedding" in error_str:
                # If the error is about invalid arguments, the function exists
                print("match_site_pages RPC function exists!")
                return True
            else:
                print(f"Unknown error checking for match_site_pages: {e}")
                return False
    except Exception as e:
        print(f"Error checking Supabase functions: {e}")
        return False

async def main():
    """Main function."""
    await check_railway_status()
    
    # Check if the match_documents RPC function exists
    await check_supabase_functions()
    
    if "--compare" in sys.argv or "-c" in sys.argv:
        await compare_files()
    elif "--drive" in sys.argv or "-d" in sys.argv:
        await list_actual_gdrive_files()
    else:
        await list_gdrive_files()
    
    if "--force-refresh" in sys.argv:
        await force_refresh()
    
    if "--delete" in sys.argv:
        file_index = sys.argv.index("--delete")
        if file_index + 1 < len(sys.argv):
            file_name = sys.argv[file_index + 1]
            await manually_delete_file(file_name)

    print("\nTo compare with Google Drive, run: python check_database.py --compare")
    print("To view only Google Drive files, run: python check_database.py --drive")
    print("To force a refresh, run: python check_database.py --force-refresh")
    print("To delete a file, run: python check_database.py --delete \"file name\"")

if __name__ == "__main__":
    asyncio.run(main())
