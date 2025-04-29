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

# Create directories and init files if they don't exist
os.makedirs('utils', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('processors', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Make sure init files exist
for directory in ['utils', 'database', 'processors']:
    init_file = os.path.join(directory, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Package initialization')

# Import from project modules
try:
    from config import (
        SUPABASE_URL, SUPABASE_KEY, CHARITIES_TABLE, RAG_CONTEXTS_TABLE,
        APP_TITLE, APP_SUBTITLE, DEFAULT_BATCH_SIZE, INCLUDE_METADATA
    )
except ImportError:
    # Fallback defaults if config is missing
    st.error("Config file not found. Using default values.")
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
    CHARITIES_TABLE = "charities"
    RAG_CONTEXTS_TABLE = "rag_contexts"
    APP_TITLE = "Charity Data RAG Assistant"
    APP_SUBTITLE = "Convert charity database to LLM-friendly paragraphs"
    DEFAULT_BATCH_SIZE = 100
    INCLUDE_METADATA = True

try:
    from database.supabase_client import SupabaseClient
    from processors.text_converter import TextConverter
    from utils.helpers import format_time, save_text_to_file, generate_filename
except ImportError:
    st.error("Required modules not found. Make sure all files are in the correct directories.")
    
    # Define fallback helper functions
    def format_time(seconds):
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.0f}s"
            
    def generate_filename(prefix="charity_rag", extension="txt"):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"
        
    def save_text_to_file(text, filename):
        if not filename.endswith('.txt'):
            filename += '.txt'
        os.makedirs('outputs', exist_ok=True)
        file_path = os.path.join('outputs', filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return file_path

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
    .stButton>button {
        width: 100%;
    }
    .sidebar .stButton>button {
        margin-bottom: 10px;
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
if "table_name" not in st.session_state:
    st.session_state.table_name = CHARITIES_TABLE
if "show_diagnostics" not in st.session_state:
    st.session_state.show_diagnostics = False
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

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
                         include_metadata: bool = INCLUDE_METADATA,
                         table_name: str = None) -> tuple:
    """
    Convert charity data to paragraphs for RAG.
    
    Args:
        batch_size: Size of each processing batch
        include_metadata: Whether to include metadata in the output
        table_name: Optional table name override
        
    Returns:
        tuple: (success, message, elapsed_time)
    """
    start_time = time.time()
    
    if table_name is None:
        table_name = st.session_state.table_name
    
    try:
        # Fetch data from Supabase
        df = st.session_state.supabase_client.fetch_all_charities(table_name=table_name)
        
        # Debug: Display the first few rows and column names
        if st.session_state.debug_mode:
            st.write("Column names:", df.columns.tolist())
            if not df.empty:
                st.write("Sample data (first 2 rows):")
                st.write(df.head(2))
            else:
                st.warning(f"No data found in the '{table_name}' table")
        
        if df.empty:
            return False, f"No data found in the '{table_name}' table. Please check your database or try another table name.", 0
        
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
    current_table = st.session_state.table_name
    st.write(f"#### 3. Testing access to '{current_table}' table")
    try:
        response = client.table(current_table).select('*').limit(1).execute()
        if response.data:
            st.success(f"✅ Successfully queried '{current_table}' table. Found data!")
            st.json(response.data[0])
        else:
            st.warning(f"⚠️ '{current_table}' table exists but contains no data")
    except Exception as e:
        st.error(f"❌ Error accessing '{current_table}' table: {str(e)}")
        
        # Try alternate table names
        alternate_tables = ['charities', 'charities_gov', 'testv2.charities', 'testv2.charities_gov', 'public.charities']
        found_alt_table = False
        
        for alt_table in alternate_tables:
            # Skip the current table name that we already tried
            if alt_table == current_table:
                continue
                
            st.write(f"Trying alternate table name: `{alt_table}`")
            try:
                response = client.table(alt_table).select('*').limit(1).execute()
                if response.data:
                    st.success(f"✅ Successfully queried '{alt_table}' table. Found data!")
                    st.json(response.data[0])
                    st.info(f"💡 Update your table name to '{alt_table}'")
                    found_alt_table = True
                    
                    # Add a button to update the table name in session state
                    if st.button(f"Use '{alt_table}' as table name"):
                        st.session_state.table_name = alt_table
                        st.success(f"Table name updated to '{alt_table}'")
                        st.experimental_rerun()
                    
                    break
                else:
                    st.warning(f"⚠️ '{alt_table}' table exists but contains no data")
            except Exception as alt_e:
                st.warning(f"⚠️ Could not access '{alt_table}': {str(alt_e)}")
        
        if not found_alt_table:
            st.error("❌ No valid tables found. Please create a 'charities' table in your database.")
            st.info("To create a new table, follow these steps:")
            st.markdown("""
            1. Go to the Supabase dashboard
            2. Click on "Table Editor" in the left navigation
            3. Click "New Table"
            4. Name it "charities"
            5. Add the following columns:
               - id (type: int8, primary key)
               - "S/N" (type: text)
               - "Name of Organisation" (type: text)
               - "Type" (type: text)
               - "UEN" (type: text)
               - "IPC Period" (type: text)
               - "Sector" (type: text)
               - "Classification" (type: text)
               - "Activities" (type: text)
            6. Click "Save" to create the table
            """)
    
    # Test 4: Print configuration
    st.write("#### 4. Current Configuration")
    st.write(f"CHARITIES_TABLE = '{CHARITIES_TABLE}'")
    st.write(f"Current table name = '{st.session_state.table_name}'")
    st.write(f"RAG_CONTEXTS_TABLE = '{RAG_CONTEXTS_TABLE}'")
    
    # Test 5: Check if data was imported
    st.write("#### 5. Check if CSV was imported")
    try:
        # Try to count rows in the charities table
        response = client.table(current_table).select('*', count='exact').execute()
        row_count = response.count if hasattr(response, 'count') else "unknown"
        st.write(f"Rows in '{current_table}' table: {row_count}")
        
        if row_count == 0 or row_count == "unknown":
            st.warning(f"⚠️ No data found in the '{current_table}' table. Did you import the CSV?")
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

def create_csv_upload_ui():
    """Create UI for CSV upload as an alternative way to get data"""
    st.subheader("CSV Upload (Alternative)")
    
    uploaded_file = st.file_uploader("Upload a CSV file if your database is empty", type="csv")
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Show columns
            st.write("Column names:", df.columns.tolist())
            
            # Show sample data
            st.write("Sample data (first 3 rows):")
            st.dataframe(df.head(3))
            
            # Store in session state for later use
            st.session_state.csv_data = df
            
            # Button to use this data
            if st.button("Use this CSV data"):
                if "converted_text" in st.session_state:
                    # Clear previous result
                    st.session_state.converted_text = None
                
                st.info("Processing CSV data...")
                # Convert to paragraph text
                text = TextConverter.format_for_rag(
                    df, 
                    include_metadata=st.session_state.get("include_metadata", INCLUDE_METADATA), 
                    batch_size=st.session_state.get("batch_size", DEFAULT_BATCH_SIZE)
                )
                
                # Store the result in session state
                st.session_state.converted_text = text
                st.success("CSV data processed successfully!")
                st.experimental_rerun()
        
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            st.session_state.csv_data = None

# Main application UI
def main():
    # Title and description
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    
    # Check for Supabase credentials
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ Supabase credentials are missing. Please set them in your Streamlit secrets or environment variables.")
        st.info("For local development, you can create a .streamlit/secrets.toml file with the following:")
        st.code("""
        SUPABASE_URL = "your-supabase-url"
        SUPABASE_KEY = "your-supabase-key"
        """)
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Connection status
        st.subheader("Database Connection")
        if st.button("Initialize Database Connection"):
            with st.spinner("Connecting to Supabase..."):
                if initialize_supabase():
                    st.success("Connected to Supabase!")
                else:
                    st.error("Failed to connect. See main panel for details.")
        
        # Table name input (editable)
        if "table_name" in st.session_state:
            custom_table = st.text_input("Table Name", value=st.session_state.table_name)
            if custom_table != st.session_state.table_name:
                st.session_state.table_name = custom_table
                st.info(f"Table name updated to '{custom_table}'")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        
        # Diagnostics tool
        st.subheader("Diagnostics")
        if st.button("Run Database Diagnostics"):
            if st.session_state.supabase_client is None:
                st.error("Please initialize the database connection first")
            else:
                st.session_state.show_diagnostics = True
                
        # Settings
        st.subheader("Settings")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=DEFAULT_BATCH_SIZE)
        include_metadata = st.checkbox("Include Metadata", value=INCLUDE_METADATA)
        
        # Process action
        st.subheader("Actions")
        if st.button("Convert to RAG Format"):
            if st.session_state.supabase_client is None:
                st.error("Please initialize the database connection first")
            else:
                st.session_state.processing = True
        
        # Download
        if st.session_state.converted_text is not None:
            st.subheader("Save Results")
            download_filename = st.text_input("Filename", value=generate_filename())
            if st.button("Save to File"):
                success, message, path = save_converted_text(st.session_state.converted_text, download_filename)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Main panel
    if st.session_state.processing:
        with st.spinner("Processing charity data..."):
            success, message, elapsed_time = convert_to_paragraphs(
                batch_size=batch_size,
                include_metadata=include_metadata,
                table_name=st.session_state.table_name
            )
            st.session_state.processing = False
            
            if success:
                st.success(f"{message} in {format_time(elapsed_time)}")
            else:
                st.error(message)
                # Show the CSV upload option as a fallback
                create_csv_upload_ui()
    
    # Show the result if available
    if st.session_state.converted_text is not None:
        st.subheader("Generated RAG Context Text")
        
        # Show stats
        text_len = len(st.session_state.converted_text)
        num_paragraphs = st.session_state.converted_text.count('\n\n') + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Text Length", f"{text_len:,} chars")
        with col2:
            st.metric("Paragraphs", f"{num_paragraphs:,}")
        with col3:
            if st.session_state.last_process_time:
                st.metric("Process Time", format_time(st.session_state.last_process_time))
        
        # Show preview
        with st.expander("Preview Text", expanded=False):
            preview_length = min(10000, len(st.session_state.converted_text))
            st.markdown(f"```\n{st.session_state.converted_text[:preview_length]}\n```")
            if preview_length < len(st.session_state.converted_text):
                st.info(f"Showing first {preview_length:,} of {len(st.session_state.converted_text):,} characters.")
    
    # Show diagnostics if requested
    if st.session_state.get("show_diagnostics", False):
        with st.container():
            st.markdown("<div class='diagnostic-box'>", unsafe_allow_html=True)
            run_database_diagnostics()
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Hide Diagnostics"):
                st.session_state.show_diagnostics = False
                st.experimental_rerun()
    
    # If no client is initialized yet, show instructions
    if st.session_state.supabase_client is None:
        st.info("👈 Please start by initializing the database connection in the sidebar")
        
        # Add a section about uploading CSV directly
        st.markdown("---")
        create_csv_upload_ui()

# Run the application
if __name__ == "__main__":
    main()