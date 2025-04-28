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
        # Updated to match the actual column names in the database
        name = row.get('name', '')
        # Since we don't have these fields, either map them to other columns or leave as empty
        org_type = row.get('charity_type', 'charitable')  # Default to 'charitable' if not found
        uen = str(row.get('charity_id', ''))  # Using charity_id as a substitute for UEN
        description = row.get('description', '')  # Using description field for activities
        
        # Handle case where organization name is missing
        if not name:
            # Check if UEN exists and is not empty
            if uen:
                paragraph = f"An unnamed {org_type.lower()} organization with ID {uen}. "
            else:
                paragraph = f"An unnamed {org_type.lower()} organization with no ID specified. "
        else:
            # Check if UEN exists and is not empty
            if uen:
                paragraph = f"{name} is a {org_type.lower()} organization with ID {uen}. "
            else:
                paragraph = f"{name} is a {org_type.lower()} organization. "
        
        # Add description if available
        if description:
            paragraph += f"Description: {description}"
        
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
        
        # Adjusted to use available columns in the actual data
        header = f"CHARITY DATABASE OVERVIEW\n\n"
        header += f"This document contains information about {num_orgs} charitable organizations. "
        
        # Get unique names for a more meaningful summary if available
        if 'name' in df.columns:
            unique_names = df['name'].dropna().unique()
            if len(unique_names) > 0:
                sample_names = [str(n) for n in unique_names[:5] if n]  # Get up to 5 examples
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
        print(f"First row sample: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
        
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