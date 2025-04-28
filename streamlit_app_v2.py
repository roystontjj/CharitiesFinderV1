"""
Main Streamlit application for Charity RAG Assistant.
"""
import streamlit as st
import pandas as pd
import time
import os
from typing import Dict, Any, List

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

def initialize_supabase():
    """Initialize Supabase client connection."""
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            st.session_state.supabase_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
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
        st.write("Sample data (first 2 rows):")
        st.write(df.head(2))
        
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

def main():
    """Main application function."""
    st.title(APP_TITLE)
    st.markdown(APP_SUBTITLE)
    
    # Initialize Supabase client if not done already
    if not st.session_state.supabase_client:
        initialize_supabase()
    
    # Sidebar with configuration options
    with st.sidebar:
        st.header("Settings")
        
        # Connection status
        if st.session_state.supabase_client:
            st.success("Connected to Supabase")
        else:
            st.error("Not connected to Supabase")
            if st.button("Connect to Supabase"):
                initialize_supabase()
        
        st.subheader("Conversion Settings")
        batch_size = st.number_input(
            "Batch Size", 
            min_value=10, 
            max_value=1000, 
            value=DEFAULT_BATCH_SIZE,
            help="Number of records to process in each batch"
        )
        
        include_metadata = st.checkbox(
            "Include Metadata Header", 
            value=INCLUDE_METADATA,
            help="Include a metadata header with dataset information"
        )
        
        # Display some info about the last process
        if st.session_state.last_process_time:
            st.info(f"Last processing time: {format_time(st.session_state.last_process_time)}")
    
    # Main content area
    if not st.session_state.supabase_client:
        st.warning("Please connect to Supabase using the sidebar button.")
        return
    
    # Show database info
    try:
        columns = st.session_state.supabase_client.fetch_table_columns(CHARITIES_TABLE)
        with st.expander("Database Information"):
            st.write(f"Table: {CHARITIES_TABLE}")
            st.write(f"Columns: {', '.join(columns)}")
    except Exception as e:
        st.error(f"Error fetching database information: {e}")
    
    # Conversion button
    col1, col2 = st.columns([1, 3])
    with col1:
        convert_button = st.button(
            "Convert to Paragraphs", 
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing
        )
    
    # Process data when button is clicked
    if convert_button:
        st.session_state.processing = True
        
        with st.spinner("Converting charities data to paragraph text..."):
            success, message, elapsed_time = convert_to_paragraphs(
                batch_size=batch_size, 
                include_metadata=include_metadata
            )
        
        if success:
            st.success(f"{message} in {format_time(elapsed_time)}")
        else:
            st.error(message)
        
        st.session_state.processing = False
    
    # Display the converted text if available
    if st.session_state.converted_text:
        st.subheader("Converted Text for RAG")
        
        # Show a sample of the text
        with st.expander("Preview", expanded=True):
            preview_text = st.session_state.converted_text[:2000]
            if len(st.session_state.converted_text) > 2000:
                preview_text += "..."
            st.markdown(
                f"""<div class="result-box">{preview_text}</div>""", 
                unsafe_allow_html=True
            )
        
        # Download options
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Save to File", use_container_width=True):
                filename = generate_filename()
                success, message, file_path = save_converted_text(
                    st.session_state.converted_text, 
                    filename
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            # Download button
            st.download_button(
                "Download Text",
                data=st.session_state.converted_text,
                file_name=generate_filename(),
                mime="text/plain",
                use_container_width=True
            )
        
        # Save to RAG contexts table
        with st.expander("Save to RAG Contexts Table"):
            table_name = st.text_input("Table Name", value=RAG_CONTEXTS_TABLE)
            
            metadata = {
                "source": CHARITIES_TABLE,
                "record_count": len(st.session_state.converted_text.split("\n\n")),
                "include_metadata": include_metadata,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.json(metadata)
            
            if st.button("Save to Database"):
                try:
                    success = st.session_state.supabase_client.save_text_to_table(
                        st.session_state.converted_text,
                        table_name=table_name,
                        metadata=metadata
                    )
                    
                    if success:
                        st.success(f"Successfully saved to {table_name} table")
                    else:
                        st.error(f"Failed to save to {table_name} table")
                except Exception as e:
                    st.error(f"Error saving to database: {e}")
    
    # Information about the app
    with st.expander("About RAG Framework"):
        st.markdown("""
        ### Retrieval-Augmented Generation (RAG)
        
        RAG is a framework that enhances large language models (LLMs) by providing them with 
        additional context from external data sources. The process involves:
        
        1. **Retrieval**: Fetching relevant information from a database or knowledge base
        2. **Augmentation**: Combining the retrieved information with the user's query
        3. **Generation**: Using an LLM to generate a response based on the combined context
        
        By converting structured data (like database tables) into natural language paragraphs,
        we make it easier for the LLM to understand and utilize the information effectively.
        """)

if __name__ == "__main__":
    main()