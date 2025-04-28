"""
Streamlit app for debugging charity data conversion
"""
import streamlit as st
import pandas as pd
import os
from text_converter import TextConverter
from supabase_client import SupabaseClient

def initialize_supabase():
    """Initialize Supabase client with credentials from environment or UI"""
    # Get credentials from environment variables or Streamlit secrets
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    
    # Allow user to input credentials if not found in environment
    if not supabase_url or not supabase_key:
        with st.expander("Supabase Connection Settings"):
            supabase_url = st.text_input("Supabase URL", supabase_url)
            supabase_key = st.text_input("Supabase Key", supabase_key, type="password")
    
    if supabase_url and supabase_key:
        try:
            client = SupabaseClient(supabase_url, supabase_key)
            return client
        except Exception as e:
            st.error(f"Failed to initialize Supabase client: {e}")
            return None
    else:
        st.warning("Supabase credentials not provided. Use CSV upload instead.")
        return None

def load_csv_data(uploaded_file=None, csv_path=None):
    """Load data from CSV file"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded uploaded CSV with {len(df)} rows")
            return df
        elif csv_path:
            df = pd.read_csv(csv_path)
            st.success(f"✅ Successfully loaded CSV from path with {len(df)} rows")
            return df
        else:
            st.warning("No CSV file provided")
            return None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def fetch_supabase_data(client, limit=100, filter_column=None, filter_value=None):
    """Fetch data from Supabase"""
    try:
        if filter_column and filter_value:
            df = client.fetch_charities_with_filter(filter_column, filter_value)
        else:
            if limit:
                df = client.fetch_charities_with_limit(limit)
            else:
                df = client.fetch_all_charities()
        
        st.success(f"✅ Successfully fetched {len(df)} records from Supabase")
        return df
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return None

def inspect_dataframe(df):
    """Inspect the DataFrame to help debug issues"""
    if df is None or len(df) == 0:
        st.warning("No data to inspect")
        return
    
    st.subheader("Data Overview")
    
    # Display basic info
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Show column names and types
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns]
    })
    st.write("Column Information:")
    st.dataframe(col_info)
    
    # Check for expected columns
    expected_columns = [
        'Name of Organisation', 'Type', 'UEN', 'IPC Period', 
        'Sector', 'Classification', 'Activities'
    ]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"❗ Missing expected columns: {', '.join(missing_cols)}")
        
        # Suggest column mappings if names are similar but not exact
        all_cols_lower = {col.lower(): col for col in df.columns}
        for missing in missing_cols:
            missing_lower = missing.lower()
            if missing_lower in all_cols_lower:
                st.info(f"Column '{all_cols_lower[missing_lower]}' might match '{missing}'")
    else:
        st.success("✅ All expected columns found")
    
    # Show sample data
    st.subheader("Sample Data (First 5 rows)")
    st.dataframe(df.head())

def debug_text_conversion(df):
    """Debug text conversion by showing intermediate steps"""
    if df is None or len(df) == 0:
        st.warning("No data to convert")
        return
    
    st.subheader("Text Conversion Debugging")
    
    # Test single row conversion
    st.write("Single Row Conversion Test:")
    sample_row_idx = st.slider("Select row to test", 0, min(10, len(df)-1), 0)
    sample_row = df.iloc[sample_row_idx].to_dict()
    
    # Display row data
    st.json(sample_row)
    
    # Extract fields with fallbacks to empty strings for missing data
    name = sample_row.get('Name of Organisation', '')
    org_type = sample_row.get('Type', '')
    uen = sample_row.get('UEN', '')
    ipc_period = sample_row.get('IPC Period', '')
    sector = sample_row.get('Sector', '')
    classification = sample_row.get('Classification', '')
    activities = sample_row.get('Activities', '')
    
    # Show extracted fields
    st.write("Extracted Fields:")
    extracted = {
        "Name": name,
        "Type": org_type,
        "UEN": uen,
        "IPC Period": ipc_period,
        "Sector": sector,
        "Classification": classification,
        "Activities": activities
    }
    st.json(extracted)
    
    # Generate and show paragraph
    paragraph = TextConverter.charity_to_paragraph(sample_row)
    st.write("Generated Paragraph:")
    st.write(paragraph)
    
    # Test batch conversion with a few rows
    st.subheader("Batch Conversion Test")
    batch_size = st.slider("Number of rows to test", 1, min(20, len(df)), 5)
    test_df = df.head(batch_size)
    
    paragraphs = TextConverter.charities_df_to_paragraphs(test_df)
    
    st.write(f"Generated {len(paragraphs)} paragraphs:")
    for i, p in enumerate(paragraphs):
        st.text(f"[{i+1}] {p}")
    
    # Generate full RAG text with small sample
    if st.button("Generate Full RAG Text (Sample)"):
        with st.spinner("Generating RAG text..."):
            rag_text = TextConverter.format_for_rag(test_df)
            st.text_area("RAG Text Output:", rag_text, height=400)

def main():
    st.set_page_config(page_title="Charity Data Converter", page_icon="📊", layout="wide")
    
    st.title("Charity Data Converter Debug Tool")
    st.write("This tool helps debug the conversion of charity data to RAG text format.")
    
    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["CSV Upload", "Supabase", "Debug & Convert"])
    
    # Tab 1: CSV Upload
    with tab1:
        st.header("Load Data from CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        csv_path = st.text_input("Or enter CSV file path", "charities_rows 2.csv")
        
        if st.button("Load CSV"):
            df = load_csv_data(uploaded_file, csv_path if not uploaded_file else None)
            if df is not None:
                st.session_state['data'] = df
                st.success("✅ Data loaded successfully. Go to 'Debug & Convert' tab.")
    
    # Tab 2: Supabase
    with tab2:
        st.header("Load Data from Supabase")
        client = initialize_supabase()
        
        if client:
            col1, col2 = st.columns(2)
            with col1:
                use_filter = st.checkbox("Use filter")
                limit = st.number_input("Limit number of records", min_value=1, max_value=1000, value=100)
            
            with col2:
                if use_filter:
                    columns = client.fetch_table_columns()
                    filter_column = st.selectbox("Filter column", options=columns if columns else ["id"])
                    filter_value = st.text_input("Filter value")
                else:
                    filter_column = None
                    filter_value = None
            
            if st.button("Fetch Data"):
                df = fetch_supabase_data(client, limit if limit > 0 else None, filter_column, filter_value)
                if df is not None:
                    st.session_state['data'] = df
                    st.success("✅ Data fetched successfully. Go to 'Debug & Convert' tab.")
    
    # Tab 3: Debug & Convert
    with tab3:
        st.header("Debug & Convert Data")
        
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            # Inspect data
            inspect_dataframe(df)
            
            # Debug text conversion
            debug_text_conversion(df)
            
            # Generate and save RAG text
            st.subheader("Generate RAG Text")
            col1, col2 = st.columns(2)
            with col1:
                include_metadata = st.checkbox("Include metadata header", value=True)
                batch_size = st.number_input("Batch size", min_value=10, max_value=1000, value=100)
            
            with col2:
                max_rows = st.number_input("Max rows to process", min_value=1, max_value=len(df), value=min(100, len(df)))
                process_df = df.head(max_rows)
            
            if st.button("Generate Complete RAG Text"):
                with st.spinner("Generating RAG text..."):
                    rag_text = TextConverter.format_for_rag(process_df, include_metadata, batch_size)
                    st.session_state['rag_text'] = rag_text
                    
                    # Show statistics
                    paragraphs = rag_text.split("\n\n")
                    word_count = len(rag_text.split())
                    st.success(f"✅ Generated RAG text with {len(paragraphs)} paragraphs and {word_count} words")
                    
                    # Show preview
                    with st.expander("RAG Text Preview"):
                        st.text_area("RAG Text:", rag_text, height=400)
            
            # Save to Supabase if available
            if 'rag_text' in st.session_state and client:
                if st.button("Save RAG Text to Supabase"):
                    metadata = {
                        "record_count": len(process_df),
                        "generated_date": pd.Timestamp.now().isoformat()
                    }
                    
                    success = client.save_text_to_table(
                        st.session_state['rag_text'], 
                        table_name='rag_contexts',
                        metadata=metadata
                    )
                    
                    if success:
                        st.success("✅ RAG text saved to Supabase successfully")
                    else:
                        st.error("❌ Failed to save RAG text to Supabase")
        else:
            st.info("Please load data from CSV or Supabase first")

if __name__ == "__main__":
    main()