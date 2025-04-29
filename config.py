"""
Configuration settings for the application.
"""
import streamlit as st

# Get API keys from Streamlit secrets
try:
    # Supabase configuration
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    # Google Gemini configuration
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.warning(f"Could not load secrets. Please make sure they are configured in Streamlit's settings: {e}")
    SUPABASE_URL = ""
    SUPABASE_KEY = ""
    GEMINI_API_KEY = ""

# Database table names - simple setup without schema prefixes
CHARITIES_TABLE = "charities"  # New simplified table name in the default schema
RAG_CONTEXTS_TABLE = "rag_contexts"  # Using default schema

# App settings
APP_TITLE = "Charity Data RAG Assistant"
APP_SUBTITLE = "Convert charity database to LLM-friendly paragraphs"

# RAG settings
DEFAULT_BATCH_SIZE = 100
INCLUDE_METADATA = True