import os
import subprocess
import sys

# Get the port from environment variable or use default
port = os.environ.get("PORT", "8501")

# Build the Streamlit command
command = [
    "streamlit", "run", "streamlit_ui.py", 
    "--server.address=0.0.0.0", 
    f"--server.port={port}"
]

# Print the command for debugging
print(f"Starting Streamlit with command: {' '.join(command)}")

# Execute the command
subprocess.run(command)
