"""
Data processing and cleaning utilities for links and embeddings data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from config import (
    INVALID_URL_PATTERNS, INVALID_EMBEDDING_WORDS, VALID_LINK_TYPES,
    VALID_STATUS_CODES, VALID_LINK_POSITIONS, COLUMNS_TO_DROP,
    POTENTIAL_URL_COLUMNS
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data cleaning and preprocessing for links and embeddings."""

    def __init__(self):
        self.invalid_url_patterns = INVALID_URL_PATTERNS
        self.invalid_embedding_words = INVALID_EMBEDDING_WORDS

    def is_valid_page(self, url: str) -> bool:
        """
        Check if a URL represents a valid page for linking.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        if pd.isna(url):
            return False

        url_str = str(url).lower()
        return not any(pattern in url_str for pattern in self.invalid_url_patterns)

    def is_valid_embedding(self, text: str) -> bool:
        """
        Check if embedding text is valid.

        Args:
            text: Embedding text to validate

        Returns:
            True if valid, False otherwise
        """
        if pd.isna(text):
            return False

        text_str = str(text).lower()

        # Check for invalid words
        if any(word in text_str for word in self.invalid_embedding_words):
            return False

        # Check for numeric content and separators
        has_numbers = any(c.isdigit() for c in text_str)
        has_separators = ',' in text_str or '.' in text_str

        return has_numbers and has_separators

    def clean_link_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process link dataset according to specified rules.

        Args:
            df: Raw link dataset

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        logger.info(f"Initial dataset shape: {df.shape}")

        # 1. Sort by Type and filter for Hyperlinks
        if 'Type' in df.columns:
            df = df.sort_values('Type')
            df = df[df['Type'].isin(VALID_LINK_TYPES)].drop('Type', axis=1)
            logger.info(f"Shape after Type filtering: {df.shape}")

        # 2. Sort by Status Code and filter for 200
        if 'Status Code' in df.columns:
            df = df.sort_values('Status Code')
            df = df[df['Status Code'].isin(VALID_STATUS_CODES)]

            columns_to_drop = ['Status Code', 'Status'] if 'Status' in df.columns else ['Status Code']
            df = df.drop(columns_to_drop, axis=1)
            logger.info(f"Shape after Status filtering: {df.shape}")

        # 3. Delete specified columns if they exist
        existing_columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(existing_columns_to_drop, axis=1)
            logger.info(f"Remaining columns: {df.columns.tolist()}")

        # 4. Process Link Position if it exists
        if 'Link Position' in df.columns:
            df = df.sort_values('Link Position')
            df = df[df['Link Position'].isin(VALID_LINK_POSITIONS)]
            logger.info(f"Shape after Link Position filtering: {df.shape}")

        # 5. Clean Source URLs
        source_col = 'Source' if 'Source' in df.columns else df.columns[0]
        df = df.sort_values(source_col)
        df = df[df[source_col].apply(self.is_valid_page)]
        logger.info(f"Shape after {source_col} URL cleaning: {df.shape}")

        # 6. Clean Destination URLs
        dest_col = 'Destination' if 'Destination' in df.columns else df.columns[1]
        df = df.sort_values(dest_col)
        df = df[df[dest_col].apply(self.is_valid_page)]
        logger.info(f"Shape after {dest_col} URL cleaning: {df.shape}")

        # 7. Process Alt Text if present
        if 'Alt Text' in df.columns and 'Anchor' in df.columns:
            df = df.sort_values('Alt Text', ascending=False)
            df.loc[df['Alt Text'].notna(), 'Anchor'] = df['Alt Text']
            df = df.drop('Alt Text', axis=1)

        # Clean up and standardize columns
        if 'Link Position' in df.columns:
            df = df.drop('Link Position', axis=1)

        if source_col != 'Source' or dest_col != 'Destination':
            df = df.rename(columns={source_col: 'Source', dest_col: 'Destination'})

        if 'Anchor' not in df.columns:
            df['Anchor'] = ''

        # Reorder columns
        final_columns = ['Source', 'Destination', 'Anchor']
        other_columns = [col for col in df.columns if col not in final_columns]
        df = df[final_columns + other_columns]

        logger.info(f"Final cleaned dataset shape: {df.shape}")
        return df

    def clean_embeddings_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess embeddings data.

        Args:
            df: Raw embeddings dataset

        Returns:
            Cleaned DataFrame or None if failed
        """
        df = df.copy()
        logger.info(f"Initial embeddings shape: {df.shape}")

        # Find the embeddings column
        embeddings_col = None
        for col in df.columns:
            if 'embeddings' in col.lower() or 'extract' in col.lower():
                embeddings_col = col
                break

        if not embeddings_col:
            logger.error("Could not find embeddings column")
            return None

        # Sort and clean embeddings
        df = df.sort_values(embeddings_col, ascending=False)
        df = df[df[embeddings_col].apply(self.is_valid_embedding)]
        logger.info(f"Shape after removing invalid embeddings: {df.shape}")

        # Filter by status code if available
        if 'Status Code' in df.columns:
            df = df[df['Status Code'].isin(VALID_STATUS_CODES)]
            logger.info(f"Shape after status code filtering: {df.shape}")

        # Determine URL column
        url_col = None
        for col in POTENTIAL_URL_COLUMNS:
            if col in df.columns:
                url_col = col
                break

        if not url_col:
            # Find first non-embeddings, non-status column
            for col in df.columns:
                if col not in [embeddings_col, 'Status Code', 'Status']:
                    url_col = col
                    break

        if not url_col:
            logger.error("Could not identify URL column")
            return None

        # Clean up columns
        cols_to_drop = [col for col in ['Status Code', 'Status'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)

        # Filter paginated URLs
        df = df[df[url_col].apply(self.is_valid_page)]
        logger.info(f"Shape after filtering paginated URLs: {df.shape}")

        # Create final dataframe
        cleaned_df = pd.DataFrame()
        cleaned_df['URL'] = df[url_col]
        cleaned_df['Embeddings'] = df[embeddings_col]

        logger.info(f"Final embeddings data shape: {cleaned_df.shape}")
        return cleaned_df

    def convert_embeddings_to_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert string embeddings to numpy arrays.

        Args:
            df: DataFrame with embeddings as strings

        Returns:
            DataFrame with embeddings as numpy arrays
        """
        def convert_embedding(embedding_str):
            try:
                # Handle different string formats
                clean_str = str(embedding_str).strip('[]').replace("'", "").replace('"', '')
                values = [float(x.strip()) for x in clean_str.split(',')]
                return np.array(values)
            except Exception as e:
                logger.error(f"Error converting embedding: {e}")
                return None

        df = df.copy()
        df['Embeddings'] = df['Embeddings'].apply(convert_embedding)

        # Remove rows with failed conversions
        df = df[df['Embeddings'].notna()]

        logger.info(f"Successfully converted {len(df)} embeddings to arrays")
        return df
