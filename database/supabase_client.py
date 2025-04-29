"""
Supabase client module for database operations with enhanced debugging.
"""
from supabase import create_client
import pandas as pd
import json
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
        
        # Debug mode
        self.debug = True
        if self.debug:
            print(f"DEBUG: Initialized Supabase client with schema={self.schema}, table={self.table_name}")
    
    def debug_print(self, message: str):
        """Print debug messages if debug mode is on"""
        if self.debug:
            print(f"DEBUG: {message}")
    
    def list_tables(self):
        """List all available tables in the database"""
        try:
            # Try to list all tables using information schema
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
            
            self.debug_print(f"Attempting to list all tables with query: {query}")
            
            try:
                # First try using rpc
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                self.debug_print(f"Found {len(response.data)} tables using RPC method")
                return pd.DataFrame(response.data)
            except Exception as e:
                self.debug_print(f"RPC method for listing tables failed: {str(e)}")
                
                # Try directly running SQL if RPC fails
                try:
                    # Some Supabase setups support direct SQL
                    response = self.client.table("pg_catalog.pg_tables").select("*").execute()
                    self.debug_print(f"Found tables using direct pg_tables query")
                    return pd.DataFrame(response.data) 
                except Exception as e2:
                    self.debug_print(f"Direct SQL for listing tables failed: {str(e2)}")
                    return pd.DataFrame()
        except Exception as e:
            self.debug_print(f"All methods for listing tables failed: {str(e)}")
            return pd.DataFrame()
            
    def fetch_all_charities(self, table_name: str = None) -> pd.DataFrame:
        """
        Fetch all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        # Before trying to fetch from the specific table, let's first try listing all tables
        self.debug_print("Attempting to list all available tables first")
        tables_df = self.list_tables()
        if not tables_df.empty:
            self.debug_print(f"Available tables: {json.dumps(tables_df.to_dict(orient='records'))}")
        else:
            self.debug_print("Could not retrieve list of tables")
        
        # Now try the specific table we're interested in
        self.debug_print(f"Now attempting to fetch from {self.full_table_name}")
        
        # Try all possible variations of the table name
        table_variations = [
            self.full_table_name,             # testv2.charities.gov
            f'"{self.schema}"."{self.table_name}"',  # "testv2"."charities.gov"
            f'{self.schema}.charities_gov',    # testv2.charities_gov (underscore)
            self.table_name,                   # charities.gov
            'charities_gov',                   # charities_gov
            'charities',                       # Original table name
            'public.charities',                # public schema
            'public.charities_gov',            # public schema with underscore
            'public."charities.gov"'           # public schema with quotes
        ]
        
        for variation in table_variations:
            self.debug_print(f"Trying table variation: {variation}")
            
            try:
                # Try table() method
                self.debug_print(f"Using table() method with: {variation}")
                response = self.client.table(variation).select('*').execute()
                if response.data:
                    self.debug_print(f"Success! Found {len(response.data)} rows using table() method")
                    return pd.DataFrame(response.data)
                else:
                    self.debug_print(f"Query succeeded but returned no data")
            except Exception as e:
                self.debug_print(f"table() method failed: {str(e)}")
                
            try:
                # Try from_() method
                self.debug_print(f"Using from_() method with: {variation}")
                # Clean any quotes for from_() method
                clean_variation = variation.replace('"', '')
                response = self.client.from_(clean_variation).select('*').execute()
                if response.data:
                    self.debug_print(f"Success! Found {len(response.data)} rows using from_() method")
                    return pd.DataFrame(response.data)
                else:
                    self.debug_print(f"Query succeeded but returned no data")
            except Exception as e:
                self.debug_print(f"from_() method failed: {str(e)}")
        
        # If all attempts fail, try to check if the table exists
        self.debug_print("All table variation attempts failed. Checking if table exists...")
        
        try:
            # Check if the table exists in the information schema
            check_query = f"""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_schema = '{self.schema}' 
                AND table_name = '{self.table_name.replace(".", "_")}'
            );
            """
            response = self.client.rpc('execute_sql', {'query': check_query}).execute()
            if response.data and response.data[0].get('exists'):
                self.debug_print(f"Table {self.schema}.{self.table_name.replace('.', '_')} exists but could not be queried")
            else:
                self.debug_print(f"Table {self.schema}.{self.table_name.replace('.', '_')} does not exist")
        except Exception as e:
            self.debug_print(f"Table existence check failed: {str(e)}")
        
        # If we get here, we couldn't find any data
        self.debug_print("All attempts to fetch data failed, returning empty DataFrame")
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
        self.debug_print(f"Attempting filtered query on {self.full_table_name} where {column}={value}")
        
        try:
            response = self.client.table(self.full_table_name).select('*').eq(column, value).execute()
            if response.data:
                self.debug_print(f"Success! Found {len(response.data)} rows with filter")
                return pd.DataFrame(response.data)
            else:
                self.debug_print("Filtered query succeeded but returned no data")
                return pd.DataFrame()
        except Exception as e:
            self.debug_print(f"Filtered query failed: {str(e)}")
            return pd.DataFrame()
    
    def fetch_charities_with_limit(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch charities with a row limit.
        
        Args:
            limit: Maximum number of rows to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing charity data
        """
        self.debug_print(f"Attempting limited query on {self.full_table_name} with limit={limit}")
        
        try:
            response = self.client.table(self.full_table_name).select('*').limit(limit).execute()
            if response.data:
                self.debug_print(f"Success! Found {len(response.data)} rows with limit")
                return pd.DataFrame(response.data)
            else:
                self.debug_print("Limited query succeeded but returned no data")
                return pd.DataFrame()
        except Exception as e:
            self.debug_print(f"Limited query failed: {str(e)}")
            return pd.DataFrame()
    
    def fetch_table_columns(self, table_name: str = None) -> List[str]:
        """
        Get the column names from a table.
        
        Args:
            table_name: Name of the table (ignored, using class defaults)
            
        Returns:
            list: List of column names
        """
        self.debug_print(f"Attempting to fetch columns from {self.full_table_name}")
        
        try:
            response = self.client.table(self.full_table_name).select('*').limit(1).execute()
            if response.data:
                columns = list(response.data[0].keys())
                self.debug_print(f"Success! Found columns: {columns}")
                return columns
            else:
                self.debug_print("Column query succeeded but returned no data")
                
                # Try to fetch columns using information schema
                try:
                    column_query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = '{self.schema}' 
                    AND table_name = '{self.table_name.replace(".", "_")}'
                    """
                    response = self.client.rpc('execute_sql', {'query': column_query}).execute()
                    if response.data:
                        columns = [row.get('column_name') for row in response.data]
                        self.debug_print(f"Found columns using information schema: {columns}")
                        return columns
                except Exception as e:
                    self.debug_print(f"Information schema column query failed: {str(e)}")
                
                return []
        except Exception as e:
            self.debug_print(f"Column fetch failed: {str(e)}")
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
        self.debug_print(f"Attempting to save text to {table_name}")
        
        try:
            # Create data with required fields
            data = {
                'content': text,
                'source_table': self.full_table_name,
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
            
            self.debug_print(f"Attempting to insert into {target_table}")
            
            try:
                # Try with schema prefix first
                response = self.client.table(target_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    self.debug_print(f"Success! Inserted data into {target_table}")
                    return True
                else:
                    self.debug_print("Insert succeeded but returned no data")
                    return False
            except Exception as e:
                self.debug_print(f"Insert with schema prefix failed: {str(e)}")
                
                # Try without schema prefix as fallback
                self.debug_print(f"Trying insert without schema prefix into {table_name}")
                response = self.client.table(table_name).insert(data).execute()
                if response.data and len(response.data) > 0:
                    self.debug_print(f"Success! Inserted data into {table_name}")
                    return True
                else:
                    self.debug_print("Insert without schema succeeded but returned no data")
                    return False
                
        except Exception as e:
            self.debug_print(f"All save attempts failed: {str(e)}")
            return False