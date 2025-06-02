#!/usr/bin/env python3
"""
Script to start a local Qdrant server.
This replaces the need for Docker to run Qdrant.
"""

import os
import subprocess
import sys
from pathlib import Path


def start_qdrant():
    # Create qdrant_data directory if it doesn't exist
    data_dir = Path("qdrant_data")
    data_dir.mkdir(exist_ok=True)

    # Get the absolute path to the data directory
    data_path = data_dir.absolute()

    # Start Qdrant server
    try:
        subprocess.run(
            [
                "qdrant",
                "--storage-path",
                str(data_path),
                "--host",
                "127.0.0.1",
                "--port",
                "6333",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error starting Qdrant server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Qdrant server not found. Please make sure it's installed.")
        print("You can install it using: pip install qdrant-server")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting local Qdrant server...")
    start_qdrant()
