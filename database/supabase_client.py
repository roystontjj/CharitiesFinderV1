"""
Supabase client module for database operations.
"""
from supabase import create_client
import pandas as pd
from typing import Optional, Dict, Any, List


class SupabaseClient:
    """
    Client for handling Supabase database operations.
    """
    def __init__(self, url: str, key: str):
        """
        Initialize Supabase client with credentials.
        
        Args:
            url: Supabase project URL
            key: Supabase API key
        """
        self.client = create_client(url, key)
        # Explicitly define the schema and table name
        self.schema = "testv2"
        self.table_name = "charities.gov"
        self.full_table_name = f"{self.schema}.{self.table_name}"
    
    def fetch_all_charities(self, table_name: str = None) -> pd.DataFrame:
        """
        Fetch all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        # Use the specific schema and table name
        try:
            # Using the table method with full schema.table name
            print(f"Attempting to fetch from {self.full_table_name}")
            response = self.client.table(self.full_table_name).select('*').execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching from {self.full_table_name}: {e}")
            try:
                # Try an alternative approach using RPC for specific schema querying
                print("Trying RPC method with schema qualification")
                query = f'SELECT * FROM "{self.schema}"."{self.table_name}"'
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                return pd.DataFrame(response.data) 
            except Exception as e2:
                print(f"RPC method failed: {e2}")
                try:
                    # Last attempt with from_ method
                    print("Trying from_ method")
                    response = self.client.from_(self.full_table_name).select('*').execute()
                    return pd.DataFrame(response.data)
                except Exception as e3:
                    print(f"All methods failed: {e3}")
                    return pd.DataFrame()
    
    def fetch_charities_with_filter(self, column: str, value: Any) -> pd.DataFrame:
        """
        Fetch charities with a specific filter.
        
        Args:
            column: Column name to filter on
            value: Value to filter by
            
        Returns:
            pd.DataFrame: DataFrame containing filtered charity data
        """
        try:
            response = self.client.table(self.full_table_name).select('*').eq(column, value).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching filtered data: {e}")
            return pd.DataFrame()
    
    def fetch_charities_with_limit(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch charities with a row limit.
        
        Args:
            limit: Maximum number of rows to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        try:
            response = self.client.table(self.full_table_name).select('*').limit(limit).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching limited data: {e}")
            return pd.DataFrame()
    
    def fetch_table_columns(self, table_name: str = None) -> List[str]:
        """
        Get the column names from a table.
        
        Args:
            table_name: Name of the table (ignored, using class defaults)
            
        Returns:
            list: List of column names
        """
        try:
            # Fetch a single row to get column names
            response = self.client.table(self.full_table_name).select('*').limit(1).execute()
            if response.data:
                return list(response.data[0].keys())
            return []
        except Exception as e:
            print(f"Error fetching columns: {e}")
            return []
    
    def save_text_to_table(self, text: str, 
                           table_name: str = 'rag_contexts', 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save generated paragraph text to a specified table for RAG.
        
        Args:
            text: The paragraph text to save
            table_name: Target table name
            metadata: Optional metadata to include
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create data with required fields
            data = {
                'content': text,
                'source_table': self.full_table_name,  # Use the qualified table name
                'is_active': True
            }
            
            # Add metadata as JSON if provided
            if metadata:
                data['metadata'] = metadata
                # Also extract record count if available
                if 'record_count' in metadata:
                    data['record_count'] = metadata['record_count']
            
            # Determine the target table with schema
            target_table = f"{self.schema}.{table_name}" if '.' not in table_name else table_name
            
            try:
                # Try with schema prefix first
                response = self.client.table(target_table).insert(data).execute()
                return len(response.data) > 0
            except Exception as e:
                print(f"Error with schema prefix: {e}")
                # Try without schema prefix as fallback
                response = self.client.table(table_name).insert(data).execute()
                return len(response.data) > 0
                
        except Exception as e:
            print(f"Error saving text to table: {e}")
            return False