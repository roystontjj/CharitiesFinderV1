"""
Text converter module to transform database tables to paragraph text for RAG.
Updated to handle exact column names from the CSV.
"""
import pandas as pd
from typing import List, Dict, Any, Optional

class TextConverter:
    """Handles conversion of structured table data to paragraph text format."""
    
    @staticmethod
    def charity_to_paragraph(row: Dict[str, Any]) -> str:
        """
        Convert a single charity record to a natural-sounding paragraph for LLM processing.
        
        Args:
            row: A dictionary representing a single charity record
            
        Returns:
            str: Formatted paragraph text optimized for vector search
        """
        # Handle both exact column names and lowercase versions for flexibility
        # Check for exact column names first
        name = row.get('Name of Organisation', row.get('name of organisation', ''))
        charity_status = row.get('Type', row.get('type', ''))
        uen = row.get('UEN', row.get('uen', ''))
        ipc_period = row.get('IPC Period', row.get('ipc period', ''))
        sector = row.get('Sector', row.get('sector', ''))
        classification = row.get('Classification', row.get('classification', ''))
        activities = row.get('Activities', row.get('activities', ''))
        
        # Start with the organization's basic identity
        if name and str(name).lower() != 'null' and str(name).strip() != '':
            # Explicitly mention the charity status
            if charity_status and str(charity_status).lower() != 'null' and str(charity_status).strip() != '':
                paragraph = f"{name} is a {charity_status} "
            else:
                paragraph = f"{name} is a charitable organization "
                
            # Include UEN if available
            if uen and str(uen).lower() != 'null' and str(uen).strip() != '':
                paragraph += f"with the UEN identifier {uen}. "
            else:
                paragraph += ". "
        else:
            # For unnamed organizations
            if charity_status and str(charity_status).lower() != 'null' and str(charity_status).strip() != '':
                paragraph = f"An unnamed {charity_status} "
            else:
                paragraph = f"An unnamed charitable organization "
                
            if uen and str(uen).lower() != 'null' and str(uen).strip() != '':
                paragraph += f"with UEN {uen}. "
            else:
                paragraph += "with no UEN specified. "
    
        # Add detailed information in a conversational flow
        details = []
        
        if ipc_period and str(ipc_period).lower() != 'null' and str(ipc_period).strip() != '':
            details.append(f"It has been granted an IPC (Institution of Public Character) status for the period {ipc_period}")
        
        if sector and str(sector).lower() != 'null' and str(sector).strip() != '':
            details.append(f"It operates within the {sector} sector")
        
        if classification and str(classification).lower() != 'null' and str(classification).strip() != '':
            details.append(f"It is classified as {classification}")
        
        # Join details with appropriate conjunctions
        if details:
            paragraph += " ".join(details) + ". "
        
        # Add activities as a concluding statement
        if activities and str(activities).lower() != 'null' and str(activities).strip() != '':
            # Clean up activities text - remove any weird formatting
            clean_activities = str(activities).replace("\n", " ").replace("\r", " ")
            while "  " in clean_activities:
                clean_activities = clean_activities.replace("  ", " ")
                
            paragraph += f"The organization's activities include: {clean_activities}."
        
        return paragraph.strip()
    
    @staticmethod
    def charities_df_to_paragraphs(df: pd.DataFrame) -> List[str]:
        """
        Convert a DataFrame of charities to a list of paragraph texts.
        
        Args:
            df: DataFrame containing charity records
            
        Returns:
            List[str]: List of paragraph texts, one for each charity
        """
        paragraphs = []
        
        # Debug: Print column names to help with troubleshooting
        print(f"Available columns in DataFrame: {df.columns.tolist()}")
        
        # Check if DataFrame is empty
        if df.empty:
            return paragraphs
            
        # Process each row in the DataFrame
        for idx, row in df.iterrows():
            try:
                # Convert row to dict first to ensure we have a consistent interface
                row_dict = row.to_dict()
                paragraph = TextConverter.charity_to_paragraph(row_dict)
                paragraphs.append(paragraph)
            except Exception as e:
                # Instead of failing, log the error and continue
                print(f"Error processing row {idx}: {e}")
                # Add a placeholder paragraph so we maintain row count
                paragraphs.append(f"Error processing charity record {idx}.")
            
        return paragraphs
    
    @staticmethod
    def concatenate_paragraphs(paragraphs: List[str], separator: str = "\n\n") -> str:
        """
        Concatenate multiple paragraphs into a single text.
        
        Args:
            paragraphs: List of paragraph texts
            separator: String to use between paragraphs (default: double newline)
            
        Returns:
            str: Concatenated text
        """
        # Filter out empty paragraphs before joining
        valid_paragraphs = [p for p in paragraphs if p]
        return separator.join(valid_paragraphs)
    
    @staticmethod
    def batch_process(df: pd.DataFrame, batch_size: int = 100) -> List[str]:
        """
        Process the DataFrame in batches to avoid memory issues with large datasets.
        
        Args:
            df: DataFrame containing charity records
            batch_size: Number of records to process in each batch
            
        Returns:
            List[str]: List of concatenated paragraph texts for each batch
        """
        results = []
        total_rows = len(df)
        
        # Handle empty DataFrame
        if total_rows == 0:
            return ["No charity records found."]
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_paragraphs = TextConverter.charities_df_to_paragraphs(batch_df)
            
            # Only add non-empty batch texts
            if batch_paragraphs:
                batch_text = TextConverter.concatenate_paragraphs(batch_paragraphs)
                results.append(batch_text)
            
        return results
    
    @staticmethod
    def create_metadata_header(df: pd.DataFrame) -> str:
        """
        Create a metadata header describing the dataset.
        
        Args:
            df: DataFrame containing charity records
            
        Returns:
            str: Metadata header text
        """
        num_orgs = len(df)
        
        # Initialize header
        header = f"CHARITY DATABASE OVERVIEW\n\n"
        header += f"This document contains information about {num_orgs} charitable organizations. "
        
        # Handle column name variations by checking both capitalized and lowercase versions
        def check_column_exists(name):
            return name in df.columns or name.lower() in df.columns
        
        def get_column_values(name):
            if name in df.columns:
                return df[name]
            elif name.lower() in df.columns:
                return df[name.lower()]
            return pd.Series()
        
        # Get sector information if available
        if check_column_exists('Sector'):
            sectors = get_column_values('Sector').dropna().unique()
            sector_list = [str(s) for s in sectors if s and str(s).lower() != 'null']
            if sector_list:
                header += f"These organizations span {len(sector_list)} sectors including: {', '.join(sector_list[:5])}"
                if len(sector_list) > 5:
                    header += ", and others. "
                else:
                    header += ". "
        
        # Get organization type information if available
        if check_column_exists('Type'):
            types = get_column_values('Type').dropna().unique()
            type_list = [str(t) for t in types if t and str(t).lower() != 'null']
            if type_list:
                header += f"The database includes various charity statuses such as: {', '.join(type_list[:5])}"
                if len(type_list) > 5:
                    header += ", and others. "
                else:
                    header += ". "
        
        # Get sample organization names
        if check_column_exists('Name of Organisation'):
            unique_names = get_column_values('Name of Organisation').dropna().unique()
            if len(unique_names) > 0:
                sample_names = [str(n) for n in unique_names[:5] if n and str(n).lower() != 'null']
                if sample_names:
                    header += f"Examples include: {', '.join(sample_names)}"
                    if len(unique_names) > 5:
                        header += ", and others."
                    else:
                        header += "."
        
        return header
    
    @staticmethod
    def format_for_rag(df: pd.DataFrame, include_metadata: bool = True, 
                    batch_size: int = 100) -> str:
        """
        Format the entire dataframe for RAG, with optional metadata header.
        
        Args:
            df: DataFrame containing charity records
            include_metadata: Whether to include a metadata header
            batch_size: Number of records to process in each batch
            
        Returns:
            str: Formatted text ready for RAG
        """
        # Debug: Print out the DataFrame info to understand its structure
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        if len(df) > 0:
            print(f"First row sample: {df.iloc[0].to_dict()}")
        
        # Make a copy of the DataFrame to avoid modification warnings
        df_copy = df.copy()
        
        # Replace NaN values with empty strings to prevent issues
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('')
        
        # Create header if requested
        header = TextConverter.create_metadata_header(df_copy) if include_metadata else ""
        
        # Process the data in batches
        batch_texts = TextConverter.batch_process(df_copy, batch_size)
        
        # Combine all parts
        if header and batch_texts:
            return header + "\n\n" + "\n\n".join(batch_texts)
        elif header:
            return header
        elif batch_texts:
            return "\n\n".join(batch_texts)
        else:
            return "No content could be generated from the provided data."