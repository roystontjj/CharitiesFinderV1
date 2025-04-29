"""
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