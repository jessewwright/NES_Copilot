"""
Streamlit Cloud deployment configuration for NES Co-Pilot Mission Control.
"""

import streamlit as st
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from app import main

# Run the main app
if __name__ == "__main__":
    main()
