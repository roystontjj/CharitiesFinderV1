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
        st.session_state.converted_text = text
        
        elapsed_time = time.time() - start_time
        st.session_state.last_process_time = elapsed_time
        
        return True, f"Successfully converted {len(df)} charity records to text", elapsed_time 
    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, f"Error during conversion: {str(e)}", elapsed_time
    
def save_converted_text(text: str, filename: str = None) -> tuple:
    """
    Save the converted text to a file.
    
    Args:
        text: Text to save
        filename: Optional filename
        
    Returns:
        tuple: (success, message, file_path)
    """
    if not filename:
        filename = generate_filename()
    
    try:
        file_path = save_text_to_file(text, filename)
        st.session_state.saved_file_path = file_path
        return True, f"Text saved to {file_path}", file_path
    except Exception as e:
        return False, f"Error saving file: {str(e)}", None

def run_database_diagnostics():
    """Run a series of diagnostics on the database connection"""
    st.write("### Database Connection Diagnostics")
    
    # Check if the Supabase client exists
    if "supabase_client" not in st.session_state or st.session_state.supabase_client is None:
        st.error("❌ No Supabase client in session state")
        return
    
    # Get the client for testing
    client = st.session_state.supabase_client.client
    
    # Test 1: Basic connection
    st.write("#### 1. Testing basic connection")
    try:
        # Try a simple query to verify connection is working
        response = client.table('_dummy_nonexistent_table').select('*').limit(1).execute()
        st.success("✅ Connection to Supabase is working (unexpected success for dummy table)")
    except Exception as e:
        error_text = str(e)
        # Check if the error mentions "relation does not exist" which is expected
        if "relation" in error_text and "does not exist" in error_text:
            st.success("✅ Connection to Supabase is working (expected error about table not found)")
        else:
            st.error(f"❌ Connection error: {error_text}")
            st.warning("Please check your Supabase URL and API key")
    
    # Test 2: List all tables
    st.write("#### 2. Listing all available tables")
    try:
        # Try to list all tables
        query = """
        SELECT 
            table_schema, 
            table_name 
        FROM 
            information_schema.tables 
        WHERE 
            table_schema NOT IN ('pg_catalog', 'information_schema')
            AND table_type = 'BASE TABLE'
        ORDER BY 
            table_schema, table_name;
        """
        
        try:
            # First try using rpc
            response = client.rpc('execute_sql', {'query': query}).execute()
            tables_df = pd.DataFrame(response.data)
            if not tables_df.empty:
                st.success(f"✅ Found {len(tables_df)} tables")
                st.dataframe(tables_df)
            else:
                st.warning("⚠️ No tables found using RPC method")
        except Exception as e:
            st.warning(f"⚠️ RPC method failed: {str(e)}")
            st.info("Trying alternative methods...")
            
            # Try directly querying pg_tables
            try:
                response = client.from_("pg_tables").select("*").execute()
                pg_tables_df = pd.DataFrame(response.data)
                if not pg_tables_df.empty:
                    # Filter out system tables
                    user_tables = pg_tables_df[~pg_tables_df['schemaname'].isin(['pg_catalog', 'information_schema'])]
                    if not user_tables.empty:
                        st.success(f"✅ Found {len(user_tables)} tables using pg_tables")
                        st.dataframe(user_tables[['schemaname', 'tablename']])
                    else:
                        st.warning("⚠️ No user tables found in pg_tables")
                else:
                    st.warning("⚠️ No tables returned from pg_tables")
            except Exception as e2:
                st.error(f"❌ All methods for listing tables failed: {str(e2)}")
    except Exception as e:
        st.error(f"❌ Error listing tables: {str(e)}")
    
    # Test 3: Check if charities table exists
    st.write("#### 3. Testing access to charities table")
    try:
        response = client.table('charities').select('*').limit(1).execute()
        if response.data:
            st.success(f"✅ Successfully queried charities table. Found data!")
            st.json(response.data[0])
        else:
            st.warning("⚠️ 'charities' table exists but contains no data")
    except Exception as e:
        st.error(f"❌ Error accessing 'charities' table: {str(e)}")
        
        # Try alternate table names
        alternate_tables = ['charities_gov', 'testv2.charities', 'testv2.charities_gov', 'public.charities']
        for alt_table in alternate_tables:
            st.write(f"Trying alternate table name: `{alt_table}`")
            try:
                response = client.table(alt_table).select('*').limit(1).execute()
                if response.data:
                    st.success(f"✅ Successfully queried '{alt_table}' table. Found data!")
                    st.json(response.data[0])
                    st.info(f"💡 Update your CHARITIES_TABLE constant in config.py to '{alt_table}'")
                    break
                else:
                    st.warning(f"⚠️ '{alt_table}' table exists but contains no data")
            except Exception as alt_e:
                st.warning(f"⚠️ Could not access '{alt_table}': {str(alt_e)}")
    
    # Test 4: Print configuration
    st.write("#### 4. Current Configuration")
    from config import CHARITIES_TABLE, RAG_CONTEXTS_TABLE
    st.write(f"CHARITIES_TABLE = '{CHARITIES_TABLE}'")
    st.write(f"RAG_CONTEXTS_TABLE = '{RAG_CONTEXTS_TABLE}'")
    
    # Test 5: Check if data was imported
    st.write("#### 5. Check if CSV was imported")
    try:
        # Try to count rows in the charities table
        response = client.table('charities').select('*', count='exact').execute()
        row_count = response.count if hasattr(response, 'count') else "unknown"
        st.write(f"Rows in charities table: {row_count}")
        
        if row_count == 0 or row_count == "unknown":
            st.warning("⚠️ No data found in the charities table. Did you import the CSV?")
            st.info("To import your CSV data:")
            st.markdown("""
            1. Go to the Supabase dashboard
            2. Navigate to Table Editor
            3. Select the "charities" table
            4. Click "Import Data" and upload your CSV file
            5. Map the columns correctly:
               - id → id
               - S/N → S/N
               - Name of Organisation → Name of Organisation
               - Type → Type
               - UEN → UEN
               - IPC Period → IPC Period
               - Sector → Sector
               - Classification → Classification
               - Activities → Activities
            """)
    except Exception as e:
        st.error(f"❌ Error checking row count: {str(e)}")