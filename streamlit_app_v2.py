"""
Main Streamlit application for Charity RAG Assistant with added diagnostics tool.
"""
import streamlit as st
import pandas as pd
import time
import os
import json
from typing import Dict, Any, List

# Force clear the cache to ensure fresh connections
st.cache_data.clear()
st.cache_resource.clear()

# Import from project modules
from config import (
    SUPABASE_URL, SUPABASE_KEY, CHARITIES_TABLE, RAG_CONTEXTS_TABLE,
    APP_TITLE, APP_SUBTITLE, DEFAULT_BATCH_SIZE, INCLUDE_METADATA
)
from database.supabase_client import SupabaseClient
from processors.text_converter import TextConverter
from utils.helpers import format_time, save_text_to_file, generate_filename

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .container {
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-height: 400px;
        overflow-y: auto;
        color: #333;  /* Add this line to ensure text color is dark */
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .diagnostic-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ffcc80;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = None
if "converted_text" not in st.session_state:
    st.session_state.converted_text = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "saved_file_path" not in st.session_state:
    st.session_state.saved_file_path = None
if "last_process_time" not in st.session_state:
    st.session_state.last_process_time = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

def initialize_supabase():
    """Initialize Supabase client connection."""
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Always recreate the client to ensure fresh connection
            st.session_state.supabase_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
            
            # Print connection details if in debug mode
            if st.session_state.debug_mode:
                st.write(f"Connecting to Supabase URL: {SUPABASE_URL[:10]}...")
            
            return True
        except Exception as e:
            st.error(f"Failed to connect to Supabase: {e}")
            return False
    else:
        st.warning("Supabase credentials not found. Please check your Streamlit secrets.")
        return False

def convert_to_paragraphs(batch_size: int = DEFAULT_BATCH_SIZE, 
                         include_metadata: bool = INCLUDE_METADATA) -> tuple:
    start_time = time.time()
    
    try:
        # Fetch data from Supabase
        df = st.session_state.supabase_client.fetch_all_charities()
        
        # Debug: Display the first few rows and column names
        st.write("Column names:", df.columns.tolist())
        if not df.empty:
            st.write("Sample data (first 2 rows):")
            st.write(df.head(2))
        else:
            st.warning("No data found in the database. Please check your table and import your CSV data.")
            
        if df.empty:
            return False, "No data found in the charities table", 0
        
        # Convert to paragraph text
        text = TextConverter.format_for_rag(
            df, 
            include_metadata=include_metadata, 
            batch_size=batch_size
        )
        
        # Store the result in session state
        st.session_stat