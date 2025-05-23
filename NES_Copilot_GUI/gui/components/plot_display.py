"""
Plot display component for the NES Co-Pilot Mission Control GUI.

This component provides reusable UI elements for displaying plots.
"""

import streamlit as st
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Union
import base64
from PIL import Image
import io

def display_image(image_path: str, caption: Optional[str] = None, width: Optional[int] = None):
    """
    Display an image from a file path.
    
    Args:
        image_path: Path to the image file.
        caption: Optional caption for the image.
        width: Optional width for the image.
    """
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}")
        return
    
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption, width=width)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def display_matplotlib_figure(fig: plt.Figure, caption: Optional[str] = None, width: Optional[int] = None):
    """
    Display a matplotlib figure.
    
    Args:
        fig: Matplotlib figure.
        caption: Optional caption for the figure.
        width: Optional width for the figure.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        image = Image.open(buf)
        st.image(image, caption=caption, width=width)
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Error displaying figure: {str(e)}")

def display_base64_image(base64_str: str, caption: Optional[str] = None, width: Optional[int] = None):
    """
    Display an image from a base64-encoded string.
    
    Args:
        base64_str: Base64-encoded string representation of the image.
        caption: Optional caption for the image.
        width: Optional width for the image.
    """
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption=caption, width=width)
    except Exception as e:
        st.error(f"Error displaying base64 image: {str(e)}")

def display_multiple_images(image_paths: List[str], captions: Optional[List[str]] = None, 
                           columns: int = 2, width: Optional[int] = None):
    """
    Display multiple images in a grid layout.
    
    Args:
        image_paths: List of paths to image files.
        captions: Optional list of captions for the images.
        columns: Number of columns in the grid.
        width: Optional width for each image.
    """
    if not image_paths:
        st.warning("No images to display.")
        return
    
    # Create a grid of columns
    cols = st.columns(columns)
    
    # Display each image in the appropriate column
    for i, image_path in enumerate(image_paths):
        col_idx = i % columns
        
        with cols[col_idx]:
            caption = captions[i] if captions and i < len(captions) else None
            display_image(image_path, caption, width)

def display_plot_with_download(fig: plt.Figure, filename: str = "plot.png", 
                              caption: Optional[str] = None, width: Optional[int] = None):
    """
    Display a matplotlib figure with a download button.
    
    Args:
        fig: Matplotlib figure.
        filename: Filename for the downloaded image.
        caption: Optional caption for the figure.
        width: Optional width for the figure.
    """
    try:
        # Display the figure
        display_matplotlib_figure(fig, caption, width)
        
        # Create a download button
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Plot",
            data=buf,
            file_name=filename,
            mime="image/png"
        )
        
        plt.close(fig)
    except Exception as e:
        st.error(f"Error displaying figure with download: {str(e)}")
