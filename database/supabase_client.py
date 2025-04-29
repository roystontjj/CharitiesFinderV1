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
    
    def fetch_all_charities(self, table_name: str = 'charities') -> pd.DataFrame:
        """
        Fetch all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        try:
            # Simple approach with default public schema
            response = self.client.table('charities').select('*').execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching charities: {e}")
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
            response = self.client.table('charities').select('*').eq(column, value).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error with filtered query: {e}")
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
            response = self.client.table('charities').select('*').limit(limit).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error with limited query: {e}")
            return pd.DataFrame()
    
    def fetch_table_columns(self, table_name: str = 'charities') -> List[str]:
        """
        Get the column names from a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            list: List of column names
        """
        try:
            response = self.client.table(table_name).select('*').limit(1).execute()
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
                'source_table': 'charities',
                'is_active': True
            }
            
            # Add metadata as JSON if provided
            if metadata:
                data['metadata'] = metadata
                # Also extract record count if available
                if 'record_count' in metadata:
                    data['record_count'] = metadata['record_count']
            
            # Simple insert into the specified table
            response = self.client.table(table_name).insert(data).execute()
            return len(response.data) > 0
        except Exception as e:
            print(f"Error saving text to table: {e}")
            return False