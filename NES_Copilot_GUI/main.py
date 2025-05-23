"""
Main Flask application for serving the NES Co-Pilot Mission Control.

This module provides a Flask wrapper around the Streamlit application.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, redirect, url_for, request, jsonify
import subprocess
import threading
import time
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flask_app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables
streamlit_process = None
streamlit_url = None
streamlit_port = 8501
streamlit_ready = False

def start_streamlit():
    """Start the Streamlit application in a separate process."""
    global streamlit_process, streamlit_ready
    
    try:
        # Path to the Streamlit app
        streamlit_app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit_app.py")
        
        # Check if the file exists
        if not os.path.exists(streamlit_app_path):
            logger.error(f"Streamlit app not found at {streamlit_app_path}")
            return
        
        # Start Streamlit
        logger.info(f"Starting Streamlit app from {streamlit_app_path}")
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", streamlit_app_path, "--server.port", str(streamlit_port), "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for Streamlit to start
        logger.info("Waiting for Streamlit to start...")
        time.sleep(5)
        streamlit_ready = True
        logger.info("Streamlit started successfully")
    except Exception as e:
        logger.error(f"Error starting Streamlit: {str(e)}")

@app.route('/')
def index():
    """Render the index page."""
    global streamlit_ready
    
    if not streamlit_ready:
        return render_template('loading.html')
    
    return render_template('index.html', streamlit_port=streamlit_port)

@app.route('/status')
def status():
    """Check if Streamlit is ready."""
    global streamlit_ready
    
    return jsonify({"ready": streamlit_ready})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

# Start Streamlit when the Flask app starts
@app.before_first_request
def before_first_request():
    """Start Streamlit before the first request."""
    global streamlit_process
    
    if streamlit_process is None:
        thread = threading.Thread(target=start_streamlit)
        thread.daemon = True
        thread.start()

if __name__ == '__main__':
    # Start Streamlit
    thread = threading.Thread(target=start_streamlit)
    thread.daemon = True
    thread.start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000)
