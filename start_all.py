#!/usr/bin/env python
import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("StartAll")

def main():
    """Start both the DriveWatcher and Streamlit app."""
    logger.info("=== Starting All Processes ===")
    
    # Start the DriveWatcher in a separate process
    logger.info("Starting DriveWatcher process...")
    watcher_process = subprocess.Popen(
        ["python", "run_watcher.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Give the DriveWatcher a moment to start
    time.sleep(2)
    
    # Log environment variables (for debugging)
    port = os.environ.get("PORT", "8501")
    logger.info(f"Using PORT: {port}")
    logger.info(f"GOOGLE_DRIVE_FOLDER_ID configured: {'Yes' if os.environ.get('GOOGLE_DRIVE_FOLDER_ID') else 'No'}")
    logger.info(f"SUPABASE_URL configured: {'Yes' if os.environ.get('SUPABASE_URL') else 'No'}")
    logger.info(f"SUPABASE_SERVICE_KEY configured: {'Yes' if os.environ.get('SUPABASE_SERVICE_KEY') else 'No'}")
    logger.info(f"GOOGLE_SERVICE_ACCOUNT_KEY configured: {'Yes' if os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY') else 'No'}")
    
    # Start a thread to monitor and log the DriveWatcher output
    def log_watcher_output():
        for line in watcher_process.stdout:
            logger.info(f"DriveWatcher: {line.strip()}")
    
    import threading
    log_thread = threading.Thread(target=log_watcher_output)
    log_thread.daemon = True
    log_thread.start()
    
    # Start Streamlit
    logger.info("Starting Streamlit app...")
    streamlit_cmd = [
        "streamlit", "run", "streamlit_ui.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}"
    ]
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")
    finally:
        # Ensure we terminate the DriveWatcher when Streamlit exits
        if watcher_process.poll() is None:
            logger.info("Terminating DriveWatcher process...")
            watcher_process.terminate()
            watcher_process.wait()
        
        logger.info("All processes stopped")

if __name__ == "__main__":
    main()
