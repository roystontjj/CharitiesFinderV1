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

# Database table names - try both formats to increase chance of success
# The actual access code has been modified to try multiple variants
CHARITIES_TABLE = "charities_gov"  # Changed from charities to charities_gov (underscore instead of dot)
RAG_CONTEXTS_TABLE = "rag_contexts"  # Changed from testv2.rag_contexts

# App settings
APP_TITLE = "Charity Data RAG Assistant"
APP_SUBTITLE = "Convert charity database to LLM-friendly paragraphs"

# RAG settings
DEFAULT_BATCH_SIZE = 100
INCLUDE_METADATA = True