import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
from agents.gdrive_watcher import DriveWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DriveWatcher")

# Load environment variables
load_dotenv()

async def main():
    """Run the DriveWatcher as a standalone process."""
    try:
        logger.info("=== Starting DriveWatcher Process ===")
        logger.info(f"GOOGLE_DRIVE_FOLDER_ID: {os.getenv('GOOGLE_DRIVE_FOLDER_ID')}")
        logger.info(f"SUPABASE_URL configured: {'Yes' if os.getenv('SUPABASE_URL') else 'No'}")
        logger.info(f"SUPABASE_SERVICE_KEY configured: {'Yes' if os.getenv('SUPABASE_SERVICE_KEY') else 'No'}")
        logger.info(f"GOOGLE_SERVICE_ACCOUNT_KEY configured: {'Yes' if os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY') else 'No'}")
        
        # Create and run the DriveWatcher
        watcher = DriveWatcher()
        await watcher.check_for_changes()
    except Exception as e:
        logger.error(f"Error in DriveWatcher: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Wait and try again to prevent immediate crash loops
        logger.info("Waiting 30 seconds before restarting...")
        await asyncio.sleep(30)
        await main()  # Restart the watcher

if __name__ == "__main__":
    logger.info("DriveWatcher process starting...")
    asyncio.run(main())
