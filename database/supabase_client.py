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
    
    def fetch_all_charities(self, table_name: str = 'charities_gov') -> pd.DataFrame:
        """
        Fetch all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        # Try multiple table name formats
        table_variations = [
            'charities_gov',      # Using underscore instead of dot
            'charities.gov',      # Using dot notation
            '"charities.gov"',    # With quotes for special characters
            'charities',          # Original table name
            'public.charities_gov',  # With schema prefix
            'public.charities.gov',  # With schema prefix and dot
            'testv2.charities_gov',  # With testv2 schema
            'testv2.charities.gov'   # With testv2 schema and dot
        ]
        
        # Try each variation
        for table_variant in table_variations:
            try:
                # Try using table method
                response = self.client.table(table_variant).select('*').execute()
                if response.data and len(response.data) > 0:
                    return pd.DataFrame(response.data)
            except Exception as e:
                pass
                
            try:
                # Try using from_ method
                response = self.client.from_(table_variant.replace('"', '')).select('*').execute()
                if response.data and len(response.data) > 0:
                    return pd.DataFrame(response.data)
            except Exception as e:
                pass
        
        # If we get here, we couldn't find data in any table variation
        # Try a final attempt with the original table
        try:
            response = self.client.from_('charities').select('*').execute()
            if response.data:
                return pd.DataFrame(response.data)
        except Exception as e:
            pass
            
        # Return empty DataFrame if all attempts failed
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
        # Try multiple table variations
        for table_name in ['charities_gov', 'charities.gov', 'charities']:
            try:
                response = self.client.table(table_name).select('*').eq(column, value).execute()
                if response.data and len(response.data) > 0:
                    return pd.DataFrame(response.data)
            except Exception:
                pass
        
        # Return empty DataFrame if all attempts failed
        return pd.DataFrame()
    
    def fetch_charities_with_limit(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch charities with a row limit.
        
        Args:
            limit: Maximum number of rows to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        # Try multiple table variations
        for table_name in ['charities_gov', 'charities.gov', 'charities']:
            try:
                response = self.client.table(table_name).select('*').limit(limit).execute()
                if response.data and len(response.data) > 0:
                    return pd.DataFrame(response.data)
            except Exception:
                pass
        
        # Return empty DataFrame if all attempts failed
        return pd.DataFrame()
    
    def fetch_table_columns(self, table_name: str = 'charities_gov') -> List[str]:
        """
        Get the column names from a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            list: List of column names
        """
        # Try different variations of the table name
        table_variants = [
            table_name,            # As provided
            table_name.replace('.', '_'),  # Replace dots with underscores
            'charities_gov',       # Hardcoded with underscore
            'charities.gov',       # Hardcoded with dot
            'charities'            # Original table name
        ]
        
        for variant in table_variants:
            try:
                response = self.client.table(variant).select('*').limit(1).execute()
                if response.data:
                    return list(response.data[0].keys())
            except Exception:
                pass
                
            try:
                response = self.client.from_(variant).select('*').limit(1).execute()
                if response.data:
                    return list(response.data[0].keys())
            except Exception:
                pass
        
        # If all attempts failed, return empty list
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
            # Find the source table that actually worked
            source_table = 'charities_gov'
            if self.fetch_all_charities('charities').shape[0] > 0:
                source_table = 'charities'
            elif self.fetch_all_charities('charities.gov').shape[0] > 0:
                source_table = 'charities.gov'
            
            # Create data with required fields
            data = {
                'content': text,
                'source_table': source_table,
                'is_active': True
            }
            
            # Add metadata as JSON if provided
            if metadata:
                data['metadata'] = metadata
                # Also extract record count if available
                if 'record_count' in metadata:
                    data['record_count'] = metadata['record_count']
            
            # Try multiple variations for the target table too
            for variant in [table_name, table_name.replace('.', '_')]:
                try:
                    response = self.client.table(variant).insert(data).execute()
                    if response.data and len(response.data) > 0:
                        return True
                except Exception:
                    pass
            
            return False
        except Exception as e:
            print(f"Error saving text to table: {e}")
            return False