"""
Configuration settings for the application.
"""
import streamlit as st
import os

# Get API keys from Streamlit secrets
try:
    # Supabase configuration
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    # Google Gemini configuration
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception as e:
    # Fallback to environment variables if secrets not available
    st.warning(f"Could not load Streamlit secrets: {e}. Attempting to load from environment variables.")
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Supabase credentials not found in secrets or environment variables.")

# Database table names - using the simple table name based on diagnostic results
CHARITIES_TABLE = "charities"  # Base table name to try first
RAG_CONTEXTS_TABLE = "rag_contexts"  # For storing generated text

# App settings
APP_TITLE = "Charity Data RAG Assistant"
APP_SUBTITLE = "Convert charity database to LLM-friendly paragraphs"

# RAG settings
DEFAULT_BATCH_SIZE = 100
INCLUDE_METADATA = True

# Add helper functions to utils directory
def setup_utils_directory():
    """Create required directories and files for the application."""
    # Ensure the utils directory exists
    os.makedirs('utils', exist_ok=True)
    
    # Create the helpers.py file if it doesn't exist
    if not os.path.exists('utils/helpers.py'):
        helpers_content = '''"""
Helper functions for the Charity RAG Assistant application.
"""
import os
import time
from datetime import datetime

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"

def generate_filename(prefix: str = "charity_rag", extension: str = "txt") -> str:
    """Generate a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def save_text_to_file(text: str, filename: str) -> str:
    """
    Save text content to a file.
    
    Args:
        text: The text content to save
        filename: Name of the file to save to
        
    Returns:
        str: The full path to the saved file
    """
    # Create 'outputs' directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Make sure filename has .txt extension
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # Full path to the file
    file_path = os.path.join('outputs', filename)
    
    # Write the text to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return file_path
'''
        # Write the helpers.py file
        with open('utils/helpers.py', 'w') as f:
            f.write(helpers_content)
    
    # Create __init__.py in utils directory to make it a proper package
    if not os.path.exists('utils/__init__.py'):
        with open('utils/__init__.py', 'w') as f:
            f.write('# Package initialization')
    
    # Create database directory and __init__.py
    os.makedirs('database', exist_ok=True)
    if not os.path.exists('database/__init__.py'):
        with open('database/__init__.py', 'w') as f:
            f.write('# Package initialization')
    
    # Create processors directory and __init__.py
    os.makedirs('processors', exist_ok=True)
    if not os.path.exists('processors/__init__.py'):
        with open('processors/__init__.py', 'w') as f:
            f.write('# Package initialization')

# Uncomment to run setup when config is imported
# setup_utils_directory()