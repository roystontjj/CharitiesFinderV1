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
    
    def fetch_all_charities(self, table_name: str = 'charities.gov') -> pd.DataFrame:
        """
        Fetch all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        try:
            # Based on your screenshot, the schema is 'testv2' and table is 'charities.gov'
            # Instead of using from_, use table() with the correct schema format
            response = self.client.table('testv2.charities.gov').select('*').execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching charities: {e}")
            # Try an alternative approach if the first one fails
            try:
                response = self.client.from_('charities.gov').select('*').execute()
                return pd.DataFrame(response.data)
            except Exception as e2:
                print(f"Second attempt error: {e2}")
                # Last resort - try without schema
                response = self.client.table('charities.gov').select('*').execute()
                return pd.DataFrame(response.data)
    
    def fetch_charities_with_filter(self, column: str, value: Any) -> pd.DataFrame:
        """
        Fetch charities with a specific filter.
        
        Args:
            column: Column name to filter on
            value: Value to filter by
            
        Returns:
            pd.DataFrame: DataFrame containing filtered charity data
        """
        response = self.client.table('charities.gov').select('*').eq(column, value).execute()
        return pd.DataFrame(response.data)
    
    def fetch_charities_with_limit(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch charities with a row limit.
        
        Args:
            limit: Maximum number of rows to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        response = self.client.table('charities.gov').select('*').limit(limit).execute()
        return pd.DataFrame(response.data)
    
    def fetch_table_columns(self, table_name: str = 'charities.gov') -> List[str]:
        """
        Get the column names from a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            list: List of column names
        """
        # Clean the table name to remove schema prefixes
        clean_table_name = table_name.split('.')[-1]
        
        # Fetch a single row to get column names
        response = self.client.table(clean_table_name).select('*').limit(1).execute()
        if response.data:
            return list(response.data[0].keys())
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
                'source_table': 'charities.gov',
                'is_active': True
            }
            
            # Add metadata as JSON if provided
            if metadata:
                data['metadata'] = metadata
                # Also extract record count if available
                if 'record_count' in metadata:
                    data['record_count'] = metadata['record_count']
                
            # Remove schema prefix if present
            clean_table_name = table_name.split('.')[-1]  # More robust approach
            response = self.client.table(clean_table_name).insert(data).execute()
            return len(response.data) > 0
        except Exception as e:
            print(f"Error saving text to table: {e}")
            return False