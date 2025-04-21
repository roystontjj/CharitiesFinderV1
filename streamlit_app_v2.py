import streamlit as st
import google.generativeai as genai
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
import time
import traceback
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Database credentials
DB_HOST = os.getenv("DB_HOST", "db.gcbtetkxkabbpwagtms.supabase.co")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "jtDyC92SeLKbSZDK")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_SCHEMA = os.getenv("DB_SCHEMA", "testv2")  # Use the new schema

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBBAOBDV4XyQzlEEl1Pl61G_YATYRmBllI")

# Page config and styling
st.set_page_config(page_title="Gemini AI Charity Explorer V2", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .chat-container {
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .user-message {
        background-color: #2b3a55;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #3e4a61;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .retrieved-context {
        background-color: #1e2b3e;
        color: #a0c0e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "Main Dashboard"
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "db_connection" not in st.session_state:
    st.session_state.db_connection = None
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None

# Database connection functions
def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    if st.session_state.db_connection is None:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
            st.session_state.db_connection = conn
            return conn
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return None
    else:
        return st.session_state.db_connection

def execute_query(query, params=None, fetch=True):
    """Execute a SQL query and return the results"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if fetch:
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        st.error(f"Query execution error: {e}")
        return None

# Initialize Gemini API
def initialize_gemini_api():
    """Initialize the Gemini API client"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        if st.session_state.gemini_model is None:
            model = genai.GenerativeModel('gemini-pro')
            st.session_state.gemini_model = model
        
        embedding_model = "models/embedding-001"
        return embedding_model
    except Exception as e:
        error_msg = f"Failed to initialize Gemini API: {str(e)}"
        if st.session_state.debug_mode:
            st.error(error_msg)
        return None

# Generate embeddings using Gemini API
def generate_embedding(text: str, model: str, task_type: str = "retrieval_query"):
    """
    Generate embedding using Gemini API
    
    Args:
        text: Text to embed
        model: Embedding model to use
        task_type: Either "retrieval_query" (for user queries) or 
                  "retrieval_document" (for database documents)
    
    Returns:
        List of floats representing the embedding vector
    """
    try:
        if not text or text.strip() == "":
            return None
        
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        
        # Return the embedding values as a list
        vector = embedding["embedding"]
        return vector
    except Exception as e:
        error_msg = f"Error generating embedding: {str(e)}"
        if st.session_state.debug_mode:
            st.error(error_msg)
        time.sleep(1)  # Add a small delay in case of rate limiting
        return None

# RAG IMPLEMENTATION
def search_charities_by_similarity(query_embedding, match_threshold=0.7, match_count=5):
    """
    Search for charities by similarity to the query embedding
    
    Args:
        query_embedding: Vector embedding of the query
        match_threshold: Minimum similarity score (0.0-1.0)
        match_count: Maximum number of results to return
        
    Returns:
        DataFrame of matching charities with similarity scores
    """
    try:
        # First, search based on embedding similarity
        embedding_query = """
        SELECT 
            c.*,
            1 - (embedding <=> %s) AS similarity
        FROM 
            testv2.charities c
        WHERE 
            embedding IS NOT NULL
            AND 1 - (embedding <=> %s) > %s
        ORDER BY 
            similarity DESC
        LIMIT %s
        """
        
        params = (query_embedding, query_embedding, match_threshold, match_count)
        results = execute_query(embedding_query, params)
        
        return results
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Error searching by similarity: {str(e)}")
        return pd.DataFrame()

def search_charities_by_text(query_text, limit=5):
    """
    Search for charities by text matching in the name or description
    
    Args:
        query_text: Search text
        limit: Maximum number of results to return
        
    Returns:
        DataFrame of matching charities
    """
    try:
        # Search based on text matching (fallback if vector search doesn't work)
        text_query = """
        SELECT *
        FROM testv2.charities
        WHERE 
            LOWER("Name of Organisation") LIKE LOWER(%s)
            OR LOWER("Type") LIKE LOWER(%s)
            OR LOWER("Sector") LIKE LOWER(%s)
            OR LOWER("Classification") LIKE LOWER(%s)
            OR LOWER("Activities") LIKE LOWER(%s)
        LIMIT %s
        """
        
        search_pattern = f"%{query_text}%"
        params = (search_pattern, search_pattern, search_pattern, search_pattern, search_pattern, limit)
        results = execute_query(text_query, params)
        
        return results
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Error searching by text: {str(e)}")
        return pd.DataFrame()

def format_results_for_rag(results: pd.DataFrame, query: str) -> str:
    """Format search results into context for RAG"""
    if results is None or results.empty:
        return "No relevant charity information found in the database for this query."
    
    formatted_context = "CHARITY DATABASE INFORMATION:\n\n"
    
    for _, charity in results.iterrows():
        name = charity.get("Name of Organisation", "Unknown Charity")
        charity_type = charity.get("Type", "Unknown Type")
        ipc_period = charity.get("IPC Period", "")
        sector = charity.get("Sector", "")
        classification = charity.get("Classification", "")
        activities = charity.get("Activities", "")
        
        formatted_context += f"CHARITY: {name}\n"
        formatted_context += f"TYPE: {charity_type}\n"
        
        if pd.notna(ipc_period) and ipc_period:
            formatted_context += f"IPC PERIOD: {ipc_period}\n"
            
        if pd.notna(sector) and sector:
            formatted_context += f"SECTOR: {sector}\n"
            
        if pd.notna(classification) and classification:
            formatted_context += f"CLASSIFICATION: {classification}\n"
            
        if pd.notna(activities) and activities:
            formatted_context += f"ACTIVITIES: {activities}\n"
            
        # Add similarity score if available
        if "similarity" in charity and pd.notna(charity["similarity"]):
            formatted_context += f"RELEVANCE: {charity['similarity']:.2f}\n"
            
        formatted_context += "\n---\n\n"
    
    return formatted_context

def create_rag_prompt(query: str, context: str) -> str:
    """Create a prompt for the LLM using RAG context"""
    return f"""
    You are a helpful charity information assistant for Singapore. Answer the user's question based on the retrieved context below.
    If the question cannot be answered based on the context provided, say so, but try to be helpful using general 
    knowledge about charities.
    
    CONTEXT:
    {context}
    
    USER QUESTION: {query}
    
    Your answer should be comprehensive and directly address the user's question.
    Include specific details from the context when available.
    Format your answer nicely with appropriate paragraphs and sections when needed.
    """

def rag_query(user_query: str, use_vector_search=True, top_k: int = 5):
    """
    Complete RAG pipeline: embedding query, retrieving context, generating answer
    
    Args:
        user_query: The user's question or query
        use_vector_search: Whether to use vector search (if False, falls back to text search)
        top_k: Number of results to retrieve
        
    Returns:
        dict: Contains the generated answer, retrieval info, and context
    """
    start_time = time.time()
    debug_info = {}
    
    try:
        results = None
        
        if use_vector_search:
            # Step 1: Generate embedding for the query
            embedding_model = initialize_gemini_api()
            query_embedding = generate_embedding(user_query, embedding_model, task_type="retrieval_query")
            
            if query_embedding:
                # Step 2: Retrieve relevant documents based on vector similarity
                results = search_charities_by_similarity(
                    query_embedding,
                    match_threshold=0.5,
                    match_count=top_k
                )
        
        # Fallback to text search if vector search returns no results
        if results is None or results.empty:
            results = search_charities_by_text(user_query, limit=top_k)
            
        # Step 3: Format retrieved documents into context
        formatted_context = format_results_for_rag(results, user_query)
        
        # Step 4: Create the RAG prompt with query and context
        rag_prompt = create_rag_prompt(user_query, formatted_context)
        
        # Step 5: Generate the final answer using Gemini
        response = st.session_state.gemini_model.generate_content(rag_prompt)
        
        # Step 6: Prepare the final response
        total_time = time.time() - start_time
        
        return {
            "answer": response.text,
            "context": formatted_context,
            "results": results,
            "total_time": total_time,
            "debug_info": debug_info
        }
        
    except Exception as e:
        error_message = f"RAG query error: {str(e)}"
        if st.session_state.debug_mode:
            error_message += f"\n{traceback.format_exc()}"
            
        return {
            "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
            "error": error_message
        }

# MAIN PAGES
def show_main_dashboard():
    """Show the main dashboard"""
    st.title("Charity Data Explorer V2")
    st.markdown("Welcome to the Singapore Charity Data Explorer with RAG (Retrieval Augmented Generation).")
    
    # Get charity stats
    conn = get_db_connection()
    if conn:
        try:
            # Count total charities
            count_query = "SELECT COUNT(*) FROM testv2.charities"
            count_df = execute_query(count_query)
            total_charities = count_df.iloc[0, 0] if not count_df.empty else 0
            
            # Count by type
            type_query = """
            SELECT "Type", COUNT(*) as count 
            FROM testv2.charities 
            GROUP BY "Type" 
            ORDER BY count DESC
            """
            type_df = execute_query(type_query)
            
            # Display stats
            st.subheader("Database Overview")
            st.metric("Total Charities", total_charities)
            
            # Display charts
            if not type_df.empty:
                st.subheader("Charities by Type")
                st.bar_chart(type_df.set_index("Type"))
                
            # Check if Sector column exists
            try:
                # Count by sector
                sector_query = """
                SELECT "Sector", COUNT(*) as count 
                FROM testv2.charities 
                WHERE "Sector" IS NOT NULL
                GROUP BY "Sector" 
                ORDER BY count DESC
                LIMIT 10
                """
                sector_df = execute_query(sector_query)
                
                if not sector_df.empty:
                    st.subheader("Top 10 Charity Sectors")
                    st.bar_chart(sector_df.set_index("Sector"))
            except:
                # Sector column might not exist
                pass
                
        except Exception as e:
            st.error(f"Error retrieving statistics: {e}")
    else:
        st.warning("Database connection not available.")
    
    # RAG query section
    st.subheader("Ask about Singapore Charities")
    with st.form(key="quick_search_form"):
        query = st.text_input("Ask a question about Singapore charities:", 
                            placeholder="e.g., Which charities focus on education?")
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("Search", use_container_width=True)
        
    if submit and query:
        with st.spinner("Searching..."):
            response = rag_query(query)
            
            st.success("Results found!")
            st.markdown("### Answer")
            st.markdown(response["answer"])
            
            with st.expander("View source data"):
                st.markdown(response["context"].replace("\n", "<br>"), unsafe_allow_html=True)

def show_charity_explorer():
    """Show the charity explorer page"""
    st.title("Charity Explorer")
    
    # Prepare search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("Search by name:", "")
    
    with col2:
        # Get all possible values for Type
        types_query = """
        SELECT DISTINCT "Type" FROM testv2.charities 
        ORDER BY "Type"
        """
        types_df = execute_query(types_query)
        types_list = ["All Types"] + list(types_df.iloc[:, 0]) if not types_df.empty else ["All Types"]
        
        selected_type = st.selectbox("Filter by type:", types_list)
    
    with col3:
        # Try to get sectors if the column exists
        try:
            sectors_query = """
            SELECT DISTINCT "Sector" FROM testv2.charities 
            WHERE "Sector" IS NOT NULL
            ORDER BY "Sector"
            """
            sectors_df = execute_query(sectors_query)
            sectors_list = ["All Sectors"] + list(sectors_df.iloc[:, 0]) if not sectors_df.empty else ["All Sectors"]
            
            selected_sector = st.selectbox("Filter by sector:", sectors_list)
        except:
            # Sector column might not exist
            selected_sector = "All Sectors"
    
    # Search button
    if st.button("Search", use_container_width=True):
        # Build the query based on filters
        base_query = "SELECT * FROM testv2.charities WHERE 1=1"
        params = []
        
        if search_query:
            base_query += ' AND LOWER("Name of Organisation") LIKE LOWER(%s)'
            params.append(f'%{search_query}%')
        
        if selected_type != "All Types":
            base_query += ' AND "Type" = %s'
            params.append(selected_type)
        
        if selected_sector != "All Sectors":
            try:
                base_query += ' AND "Sector" = %s'
                params.append(selected_sector)
            except:
                # Sector column might not exist
                pass
        
        # Add limit and order
        base_query += ' ORDER BY "Name of Organisation" LIMIT 100'
        
        # Execute the query
        results = execute_query(base_query, params)
        
        if results is not None and not results.empty:
            st.success(f"Found {len(results)} charities")
            
            # Display results
            st.dataframe(results)
            
            # Allow user to select a charity for details
            selected_charity = st.selectbox(
                "Select a charity to view details:",
                options=results["Name of Organisation"].tolist()
            )
            
            # Show details if a charity is selected
            if selected_charity:
                selected_row = results[results["Name of Organisation"] == selected_charity].iloc[0]
                
                st.subheader(f"Details for {selected_charity}")
                
                # Format the display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Type:** {selected_row.get('Type', 'N/A')}")
                    st.markdown(f"**UEN:** {selected_row.get('UEN', 'N/A')}")
                    
                with col2:
                    # Show additional columns if they exist
                    for col in ["IPC Period", "Sector", "Classification"]:
                        if col in selected_row and pd.notna(selected_row[col]):
                            st.markdown(f"**{col}:** {selected_row[col]}")
                
                # Show activities if available
                if "Activities" in selected_row and pd.notna(selected_row["Activities"]):
                    st.subheader("Activities")
                    st.write(selected_row["Activities"])
                
                # Generate AI insights
                st.subheader("AI Insights")
                if st.button("Generate insights"):
                    with st.spinner("Generating insights..."):
                        # Format the data for the AI
                        charity_data = ", ".join([f"{col}: {selected_row.get(col, 'N/A')}" 
                                               for col in selected_row.index 
                                               if pd.notna(selected_row.get(col, None))])
                        
                        prompt = f"""
                        Based on the following charity information, provide useful insights:
                        
                        {charity_data}
                        
                        Please provide:
                        1. A summary of what this charity does
                        2. The sector and type of work it focuses on
                        3. Any notable or interesting aspects
                        """
                        
                        response = st.session_state.gemini_model.generate_content(prompt)
                        st.markdown(response.text)
        else:
            st.info("No charities found matching your search criteria.")

def show_rag_chat():
    """Show the RAG chat interface"""
    st.title("💬 Charity RAG Chat Assistant")
    st.markdown("Chat with the AI assistant about Singapore charities")
    
    # Display chat history
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'><strong>You:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
                # Show retrieved context if available and debug mode is on
                if "context" in message and st.session_state.debug_mode:
                    with st.expander("View retrieved context"):
                        st.markdown(f"<div class='retrieved-context'>{message['context']}</div>", unsafe_allow_html=True)
    
    # RAG settings
    with st.expander("RAG Settings"):
        col1, col2 = st.columns(2)
        with col1:
            use_vector_search = st.checkbox("Use vector search", value=True, 
                                         help="Enable vector similarity search (if disabled, will use text search)")
        with col2:
            num_results = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
    
    # User input section
    st.subheader("Your Message")
    with st.form(key="chat_form", clear_on_submit=True):
        user_message = st.text_area("Type your message here:", height=100, key="message_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        with col2:
            if st.form_submit_button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Process the user input
    if submit_button and user_message:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Show a spinner while processing
        with st.spinner("Processing your message..."):
            try:
                rag_response = rag_query(
                    user_message,
                    use_vector_search=use_vector_search,
                    top_k=num_results
                )
                
                # Extract the answer and metadata
                answer = rag_response.get("answer", "I couldn't process your question.")
                context_html = rag_response.get("context", "").replace("\n", "<br>")
                
                # Add assistant response to chat history with context
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context_html,
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
        
        # Rerun to update the UI with the new messages
        st.rerun()

def show_database_tools():
    """Show database tools"""
    st.title("Database Tools")
    
    # Choose operation
    operation = st.radio("Select operation:", 
                       ["View Table Structure", "Run Custom Query", "Generate Embeddings"])
    
    if operation == "View Table Structure":
        # View table structure
        st.subheader("Table Structure")
        
        # Get table information
        table_query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'testv2' AND table_name = 'charities'
        ORDER BY ordinal_position
        """
        
        table_info = execute_query(table_query)
        
        if table_info is not None and not table_info.empty:
            st.dataframe(table_info)
        else:
            st.info("Unable to retrieve table structure.")
            
    elif operation == "Run Custom Query":
        # Run custom query
        st.subheader("Custom SQL Query")
        st.warning("Be careful with data modification queries!")
        
        query = st.text_area("Enter your SQL query:", height=150,
                           placeholder="SELECT * FROM testv2.charities LIMIT 10")
        
        is_select = query.strip().upper().startswith("SELECT")
        
        if st.button("Run Query"):
            with st.spinner("Executing query..."):
                if is_select:
                    results = execute_query(query)
                    
                    if results is not None:
                        st.success("Query executed successfully!")
                        st.dataframe(results)
                        
                        # Option to download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "Download results as CSV",
                            csv,
                            "query_results.csv",
                            "text/csv"
                        )
                else:
                    # Non-SELECT query
                    success = execute_query(query, fetch=False)
                    
                    if success:
                        st.success("Query executed successfully!")
                    else:
                        st.error("Error executing query.")
    
    elif operation == "Generate Embeddings":
        # Generate embeddings for charity data
        st.subheader("Generate Embeddings")
        st.info("This will generate vector embeddings for charity descriptions to enable semantic search.")
        
        # Check if pgvector extension is installed
        pgvector_query = """
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
        """
        pgvector_df = execute_query(pgvector_query)
        
        if pgvector_df is None or pgvector_df.empty:
            st.error("pgvector extension is not installed in the database.")
            
            with st.expander("SQL to install pgvector"):
                st.code("""
                -- Run this in the SQL editor of Supabase or your PostgreSQL client
                CREATE EXTENSION IF NOT EXISTS vector;
                
                -- Then add a vector column to the charities table
                ALTER TABLE testv2.charities ADD COLUMN IF NOT EXISTS embedding vector(768);
                """, language="sql")
        else:
            # Check if embedding column exists
            column_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'testv2' 
              AND table_name = 'charities' 
              AND column_name = 'embedding'
            """
            column_df = execute_query(column_query)
            
            if column_df is None or column_df.empty:
                st.warning("Embedding column does not exist in the charities table.")
                
                if st.button("Add embedding column"):
                    # Add embedding column
                    add_column_query = """
                    ALTER TABLE testv2.charities ADD COLUMN embedding vector(768);
                    """
                    success = execute_query(add_column_query, fetch=False)
                    
                    if success:
                        st.success("Embedding column added successfully!")
                    else:
                        st.error("Error adding embedding column.")
            else:
                # Count records needing embeddings
                count_query = """
                SELECT COUNT(*) 
                FROM testv2.charities 
                WHERE embedding IS NULL
                """
                count_df = execute_query(count_query)
                
                if count_df is not None and not count_df.empty:
                    records_count = count_df.iloc[0, 0]
                    
                    st.info(f"{records_count} records need embeddings generated.")
                    
                    if st.button("Generate Embeddings"):
                        # Get records without embeddings
                        query = """
                        SELECT "charity_id", "Name of Organisation", "Type", "Activities" 
                        FROM testv2.charities 
                        WHERE embedding IS NULL
                        LIMIT 50
                        """
                        records = execute_query(query)
                        
                        if records is not None and not records.empty:
                            # Initialize embedding model
                            embedding_model = initialize_gemini_api()
                            
                            # Complete
                            st.success("Embedding generation completed!")
                            
                            # Update count
                            count_df = execute_query(count_query)
                            if count_df is not None and not count_df.empty:
                                remaining = count_df.iloc[0, 0]
                                st.info(f"{remaining} records still need embeddings generated.")
                        else:
                            st.error("Failed to fetch records for embedding generation.")

def show_settings():
    """Show settings page"""
    st.title("Settings")
    
    # Database settings
    st.subheader("Database Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        db_host = st.text_input("Database Host", value=DB_HOST)
    with col2:
        db_port = st.text_input("Database Port", value=DB_PORT)
        
    col1, col2 = st.columns(2)
    with col1:
        db_name = st.text_input("Database Name", value=DB_NAME)
    with col2:
        db_schema = st.text_input("Database Schema", value=DB_SCHEMA)
        
    col1, col2 = st.columns(2)
    with col1:
        db_user = st.text_input("Database User", value=DB_USER)
    with col2:
        db_password = st.text_input("Database Password", value=DB_PASSWORD, type="password")
    
    # API settings
    st.subheader("API Settings")
    api_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    
    # Debug mode
    debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
    # Save settings
    if st.button("Save Settings"):
        # Update environment variables
        os.environ["DB_HOST"] = db_host
        os.environ["DB_PORT"] = db_port
        os.environ["DB_NAME"] = db_name
        os.environ["DB_SCHEMA"] = db_schema
        os.environ["DB_USER"] = db_user
        os.environ["DB_PASSWORD"] = db_password
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Update session state
        st.session_state.debug_mode = debug_mode
        
        # Reset connections to apply new settings
        st.session_state.db_connection = None
        st.session_state.gemini_model = None
        
        st.success("Settings saved! Reconnecting to services...")
        
        # Reinitialize connections
        get_db_connection()
        initialize_gemini_api()
        
        st.experimental_rerun()

# Main app entry point
def main():
    # Initialize first connections
    get_db_connection()
    initialize_gemini_api()
    
    # Sidebar navigation
    st.sidebar.title("Singapore Charity Explorer V2")
    
    pages = {
        "Main Dashboard": show_main_dashboard,
        "Charity Explorer": show_charity_explorer,
        "RAG Chat": show_rag_chat,
        "Database Tools": show_database_tools,
        "Settings": show_settings
    }
    
    # Let the user choose the page
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Debug mode toggle
    with st.sidebar.expander("Developer Options"):
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode
    
    # Display the selected page
    pages[selected_page]()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Google Gemini & Supabase")

if __name__ == "__main__":
    main()
progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (_, record) in enumerate(records.iterrows()):
                                # Create text for embedding
                                charity_id = record["charity_id"]
                                name = record["Name of Organisation"]
                                charity_type = record["Type"]
                                
                                # Combine fields to create rich embedding text
                                embedding_text = f"Charity: {name}. Type: {charity_type}."
                                
                                if pd.notna(record.get("Activities")) and record.get("Activities"):
                                    embedding_text += f" Activities: {record.get('Activities')}"
                                
                                # Generate embedding
                                try:
                                    embedding = generate_embedding(
                                        embedding_text, 
                                        embedding_model, 
                                        task_type="retrieval_document"
                                    )
                                    
                                    if embedding:
                                        # Update record with embedding
                                        update_query = """
                                        UPDATE testv2.charities 
                                        SET embedding = %s 
                                        WHERE "charity_id" = %s
                                        """
                                        execute_query(update_query, (embedding, charity_id), fetch=False)
                                        
                                        # Update progress
                                        progress = (i + 1) / len(records)
                                        progress_bar.progress(progress)
                                        status_text.text(f"Processed {i+1}/{len(records)}: {name}")
                                        
                                        # Add delay to avoid rate limits
                                        time.sleep(0.5)
                                    
                                except Exception as e:
                                    st.error(f"Error generating embedding for {name}: {e}")
                                    time.sleep(1)  # Longer delay on error