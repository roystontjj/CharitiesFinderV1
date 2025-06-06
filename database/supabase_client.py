"""
Supabase client module for database operations.
Updated to handle exact column names from the CSV.
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
        
        Args:
            table_name: Name of the table to query
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        try:
            # Try with the provided table name first
            response = self.client.table(table_name).select('*').execute()
            
            # Check if response has data
            if response.data:
                print(f"Successfully fetched data from '{table_name}' with {len(response.data)} records")
                return pd.DataFrame(response.data)
            else:
                print(f"Table '{table_name}' exists but contains no data")
                
            # If no data, try alternative tables
            alternate_tables = ['charities_gov', 'public.charities', 'testv2.charities']
            for alt_table in alternate_tables:
                if alt_table == table_name:
                    continue  # Skip if same as original
                
                try:
                    print(f"Trying alternate table: {alt_table}")
                    alt_response = self.client.table(alt_table).select('*').execute()
                    if alt_response.data:
                        print(f"Found data in '{alt_table}' with {len(alt_response.data)} records")
                        return pd.DataFrame(alt_response.data)
                except Exception as alt_e:
                    print(f"Error with alternate table '{alt_table}': {alt_e}")
            
            # Return empty DataFrame if all attempts fail
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching charities from '{table_name}': {e}")
            return pd.DataFrame()
    
    def fetch_charities_with_filter(self, column: str, value: Any, table_name: str = 'charities') -> pd.DataFrame:
        """
        Fetch charities with a specific filter.
        
        Args:
            column: Column name to filter on
            value: Value to filter by
            table_name: Name of the table to query
            
        Returns:
            pd.DataFrame: DataFrame containing filtered charity data
        """
        try:
            # Note: For columns with spaces, we need to use double quotes
            if ' ' in column:
                column = f'"{column}"'
                
            response = self.client.table(table_name).select('*').eq(column, value).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error with filtered query: {e}")
            return pd.DataFrame()
    
    def fetch_charities_with_limit(self, limit: int = 100, table_name: str = 'charities') -> pd.DataFrame:
        """
        Fetch charities with a row limit.
        
        Args:
            limit: Maximum number of rows to fetch
            table_name: Name of the table to query
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        try:
            response = self.client.table(table_name).select('*').limit(limit).execute()
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
            
            # If no data returned, use information schema to get columns
            query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = '{table_name}'
            """
            try:
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                if response.data:
                    return [row['column_name'] for row in response.data]
            except Exception as e:
                print(f"RPC execute_sql error: {e}")
                
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