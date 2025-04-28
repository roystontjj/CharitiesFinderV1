"""
Text converter module to transform database tables to paragraph text for RAG.
Fixed version with added debugging and error handling.
"""
import pandas as pd
from typing import List, Dict, Any, Optional

class TextConverter:
    """Handles conversion of structured table data to paragraph text format."""
    
    @staticmethod
    def charity_to_paragraph(row: Dict[str, Any]) -> str:
        """
        Convert a single charity record to paragraph text.
        
        Args:
            row: A dictionary representing a single charity record
            
        Returns:
            str: Formatted paragraph text
        """
        # Extract fields with fallbacks to empty strings for missing data
        # Check for alternative column naming formats
        name = row.get('Name of Organisation', '')
        org_type = row.get('Type', '')
        uen = row.get('UEN', '')
        ipc_period = row.get('IPC Period', '')
        sector = row.get('Sector', '')
        classification = row.get('Classification', '')
        activities = row.get('Activities', '')
        
        # Handle case where organization name is missing
        if not name:
            # Rather than skipping, provide placeholder for unnamed organizations
            paragraph = f"An unnamed {org_type.lower() if org_type else 'charitable'} organization with UEN {uen}. "
        else:
            paragraph = f"{name} is a {org_type.lower() if org_type else 'charitable'} organization with UEN {uen}. "
        
        if ipc_period:
            paragraph += f"It has an IPC period of {ipc_period}. "
            
        if sector:
            paragraph += f"The organization operates in the {sector} sector. "
            
        if classification:
            paragraph += f"It is classified as {classification}. "
            
        if activities:
            paragraph += f"The organization is involved in the following activities: {activities}."
        
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
        
        # Check if DataFrame is empty
        if df.empty:
            return paragraphs
            
        # Process each row in the DataFrame
        for idx, row in df.iterrows():
            try:
                paragraph = TextConverter.charity_to_paragraph(row)
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
        
        # Check if required columns exist
        has_sector = 'Sector' in df.columns
        has_type = 'Type' in df.columns
        
        # Get unique non-null values if columns exist
        sectors = df['Sector'].dropna().unique() if has_sector else []
        types = df['Type'].dropna().unique() if has_type else []
        
        header = f"CHARITY DATABASE OVERVIEW\n\n"
        header += f"This document contains information about {num_orgs} charitable organizations. "
        
        if len(sectors) > 0:
            sector_list = [str(s) for s in sectors if s]
            if sector_list:
                header += f"These organizations span {len(sector_list)} sectors including: {', '.join(sector_list)}. "
            
        if len(types) > 0:
            type_list = [str(t) for t in types if t]
            if type_list:
                header += f"The database includes various organization types such as: {', '.join(type_list)}."
        
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