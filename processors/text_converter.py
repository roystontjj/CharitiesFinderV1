"""
Text converter module to transform database tables to paragraph text for RAG.
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
        name = row.get('Name of Organisation', '')
        org_type = row.get('Type', '')
        uen = row.get('UEN', '')
        ipc_period = row.get('IPC Period', '')
        sector = row.get('Sector', '')
        classification = row.get('Classification', '')
        activities = row.get('Activities', '')
        
        # Create a paragraph with the information
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
        
        # Process each row in the DataFrame
        for _, row in df.iterrows():
            paragraph = TextConverter.charity_to_paragraph(row)
            paragraphs.append(paragraph)
            
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
        return separator.join(paragraphs)
    
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
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_paragraphs = TextConverter.charities_df_to_paragraphs(batch_df)
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
        sectors = df['Sector'].unique() if 'Sector' in df.columns else []
        types = df['Type'].unique() if 'Type' in df.columns else []
        
        header = f"CHARITY DATABASE OVERVIEW\n\n"
        header += f"This document contains information about {num_orgs} charitable organizations. "
        
        if len(sectors) > 0:
            sector_list = [s for s in sectors if s and not pd.isna(s)]
            if sector_list:
                header += f"These organizations span {len(sector_list)} sectors including: {', '.join(sector_list)}. "
            
        if len(types) > 0:
            type_list = [t for t in types if t and not pd.isna(t)]
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
        # Create header if requested
        header = TextConverter.create_metadata_header(df) if include_metadata else ""
        
        # Process the data in batches
        batch_texts = TextConverter.batch_process(df, batch_size)
        
        # Combine all parts
        if header:
            return header + "\n\n" + "\n\n".join(batch_texts)
        else:
            return "\n\n".join(batch_texts)