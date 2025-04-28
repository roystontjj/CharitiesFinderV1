"""
Helper utilities for the application.
"""
import os
import json
import pandas as pd
import time
from typing import Dict, Any, Optional, List, Tuple

def format_time(seconds: float) -> str:
    """
    Format time duration in seconds to a readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def save_text_to_file(text: str, filename: str, directory: str = "outputs") -> str:
    """
    Save text content to a file.
    
    Args:
        text: Text content to save
        filename: Name of the file
        directory: Directory to save the file in
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    filepath = os.path.join(directory, filename)
    
    # Write the text to the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filepath

def log_operation(operation: str, status: str, 
                 details: Optional[Dict[str, Any]] = None, 
                 error: Optional[Exception] = None) -> Dict[str, Any]:
    """
    Create a log entry for an operation.
    
    Args:
        operation: Name of the operation
        status: Status of the operation (success, error, etc.)
        details: Additional details about the operation
        error: Exception if an error occurred
        
    Returns:
        dict: Log entry
    """
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'operation': operation,
        'status': status
    }
    
    if details:
        log_entry['details'] = details
        
    if error:
        log_entry['error'] = str(error)
        
    return log_entry

def generate_filename(prefix: str = "charity_text", extension: str = "txt") -> str:
    """
    Generate a filename with timestamp.
    
    Args:
        prefix: Prefix for the filename
        extension: File extension
        
    Returns:
        str: Generated filename
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"

def chunk_text(text: str, chunk_size: int = 1000, 
              overlap: int = 200) -> List[str]:
    """
    Split text into chunks with optional overlap for better RAG retrieval.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text
        end = start + chunk_size
        chunk = text[start:end]
        
        # If not at the end of text and not at a good break point, find a good break
        if end < len(text) and text[end] not in ['.', '!', '?', '\n']:
            # Find the last sentence end within the chunk
            last_period = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'), chunk.rfind('\n'))
            if last_period > 0:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if end - overlap > start else end
    
    return chunks

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about a DataFrame.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        dict: Information about the DataFrame
    """
    info = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return info