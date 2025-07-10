"""
Data processing and cleaning utilities for links and embeddings data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from config import (
    INVALID_URL_PATTERNS, INVALID_EMBEDDING_WORDS, VALID_LINK_TYPES,
    VALID_STATUS_CODES, VALID_LINK_POSITIONS, COLUMNS_TO_DROP,
    POTENTIAL_URL_COLUMNS, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Production-grade data processor for links and embeddings data."""
    
    def __init__(self):
        self.invalid_url_patterns = INVALID_URL_PATTERNS
        self.invalid_embedding_words = INVALID_EMBEDDING_WORDS
        self.valid_link_types = VALID_LINK_TYPES
        self.valid_status_codes = VALID_STATUS_CODES
        self.valid_link_positions = VALID_LINK_POSITIONS
        self.columns_to_drop = COLUMNS_TO_DROP
        self.potential_url_columns = POTENTIAL_URL_COLUMNS
        
        # Statistics tracking
        self.processing_stats = {
            "links_processed": 0,
            "embeddings_processed": 0,
            "invalid_urls_removed": 0,
            "invalid_embeddings_removed": 0
        }
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for processing.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if pd.isna(url) or not url:
            return False
            
        url_str = str(url).lower()
        return not any(pattern in url_str for pattern in self.invalid_url_patterns)
    
    def is_valid_embedding(self, embedding: str) -> bool:
        """
        Check if an embedding string is valid.
        
        Args:
            embedding: Embedding string to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if pd.isna(embedding) or not embedding:
            return False
            
        embedding_str = str(embedding).lower()
        
        # Check for invalid words
        if any(word in embedding_str for word in self.invalid_embedding_words):
            return False
            
        # Check for numeric content and separators
        has_numbers = any(c.isdigit() for c in embedding_str)
        has_separators = ',' in embedding_str or '.' in embedding_str
        
        return has_numbers and has_separators
    
    def clean_link_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process link dataset according to production standards.
        
        Args:
            df: Raw links DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting link dataset cleaning. Initial shape: {df.shape}")
        df_cleaned = df.copy()
        
        # Track original size
        original_size = len(df_cleaned)
        
        # 1. Filter by Type if available
        if 'Type' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('Type')
            type_mask = df_cleaned['Type'].isin(self.valid_link_types)
            df_cleaned = df_cleaned[type_mask]
            logger.info(f"After Type filtering: {len(df_cleaned)} rows")
            
            # Drop Type column after filtering
            df_cleaned = df_cleaned.drop('Type', axis=1)
        
        # 2. Filter by Status Code if available
        if 'Status Code' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('Status Code')
            status_mask = df_cleaned['Status Code'].isin(self.valid_status_codes)
            df_cleaned = df_cleaned[status_mask]
            logger.info(f"After Status Code filtering: {len(df_cleaned)} rows")
            
            # Drop status-related columns
            status_columns = ['Status Code', 'Status']
            columns_to_drop_now = [col for col in status_columns if col in df_cleaned.columns]
            df_cleaned = df_cleaned.drop(columns_to_drop_now, axis=1)
        
        # 3. Drop specified columns if they exist
        columns_to_drop_now = [col for col in self.columns_to_drop if col in df_cleaned.columns]
        if columns_to_drop_now:
            df_cleaned = df_cleaned.drop(columns_to_drop_now, axis=1)
            logger.info(f"Dropped columns: {columns_to_drop_now}")
        
        # 4. Filter by Link Position if available
        if 'Link Position' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('Link Position')
            position_mask = df_cleaned['Link Position'].isin(self.valid_link_positions)
            df_cleaned = df_cleaned[position_mask]
            logger.info(f"After Link Position filtering: {len(df_cleaned)} rows")
            
            # Drop Link Position column
            df_cleaned = df_cleaned.drop('Link Position', axis=1)
        
        # 5. Identify and clean URL columns
        source_col = 'Source' if 'Source' in df_cleaned.columns else df_cleaned.columns[0]
        dest_col = 'Destination' if 'Destination' in df_cleaned.columns else df_cleaned.columns[1]
        
        # Clean Source URLs
        df_cleaned = df_cleaned.sort_values(source_col)
        source_mask = df_cleaned[source_col].apply(self.is_valid_url)
        df_cleaned = df_cleaned[source_mask]
        logger.info(f"After Source URL cleaning: {len(df_cleaned)} rows")
        
        # Clean Destination URLs
        df_cleaned = df_cleaned.sort_values(dest_col)
        dest_mask = df_cleaned[dest_col].apply(self.is_valid_url)
        df_cleaned = df_cleaned[dest_mask]
        logger.info(f"After Destination URL cleaning: {len(df_cleaned)} rows")
        
        # 6. Handle Alt Text and Anchor columns
        if 'Alt Text' in df_cleaned.columns and 'Anchor' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('Alt Text', ascending=False)
            # Use Alt Text as Anchor when available
            alt_text_mask = df_cleaned['Alt Text'].notna()
            df_cleaned.loc[alt_text_mask, 'Anchor'] = df_cleaned.loc[alt_text_mask, 'Alt Text']
            df_cleaned = df_cleaned.drop('Alt Text', axis=1)
            logger.info("Processed Alt Text into Anchor column")
        
        # 7. Standardize column names
        column_mapping = {}
        if source_col != 'Source':
            column_mapping[source_col] = 'Source'
        if dest_col != 'Destination':
            column_mapping[dest_col] = 'Destination'
            
        if column_mapping:
            df_cleaned = df_cleaned.rename(columns=column_mapping)
            logger.info(f"Renamed columns: {column_mapping}")
        
        # 8. Ensure Anchor column exists
        if 'Anchor' not in df_cleaned.columns:
            df_cleaned['Anchor'] = ''
            logger.info("Added empty Anchor column")
        
        # 9. Reorder columns
        primary_columns = ['Source', 'Destination', 'Anchor']
        other_columns = [col for col in df_cleaned.columns if col not in primary_columns]
        final_columns = primary_columns + other_columns
        df_cleaned = df_cleaned[final_columns]
        
        # 10. Remove duplicates
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['Source', 'Destination'], keep='first')
        duplicates_removed = initial_count - len(df_cleaned)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate links")
        
        # Update statistics
        self.processing_stats["links_processed"] = original_size
        self.processing_stats["invalid_urls_removed"] = original_size - len(df_cleaned)
        
        logger.info(f"Link dataset cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def clean_embeddings_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean and preprocess embeddings data.
        
        Args:
            df: Raw embeddings DataFrame
            
        Returns:
            Cleaned DataFrame or None if processing fails
        """
        logger.info(f"Starting embeddings data cleaning. Initial shape: {df.shape}")
        
        try:
            df_cleaned = df.copy()
            original_size = len(df_cleaned)
            
            # 1. Find embeddings column
            embeddings_col = None
            for col in df_cleaned.columns:
                if any(keyword in col.lower() for keyword in ['embedding', 'extract', 'vector']):
                    embeddings_col = col
                    break
            
            if not embeddings_col:
                logger.error("Could not find embeddings column")
                return None
            
            # 2. Sort and clean embeddings
            df_cleaned = df_cleaned.sort_values(embeddings_col, ascending=False)
            embeddings_mask = df_cleaned[embeddings_col].apply(self.is_valid_embedding)
            df_cleaned = df_cleaned[embeddings_mask]
            logger.info(f"After embeddings validation: {len(df_cleaned)} rows")
            
            # 3. Filter by status code if available
            if 'Status Code' in df_cleaned.columns:
                status_mask = df_cleaned['Status Code'].isin(self.valid_status_codes)
                df_cleaned = df_cleaned[status_mask]
                logger.info(f"After Status Code filtering: {len(df_cleaned)} rows")
            
            # 4. Identify URL column
            url_col = None
            for col in self.potential_url_columns:
                if col in df_cleaned.columns:
                    url_col = col
                    break
            
            if not url_col:
                # Use the first non-embeddings, non-status column
                for col in df_cleaned.columns:
                    if col not in [embeddings_col, 'Status Code', 'Status']:
                        url_col = col
                        break
            
            if not url_col:
                logger.error("Could not identify URL column")
                return None
            
            # 5. Clean URLs
            url_mask = df_cleaned[url_col].apply(self.is_valid_url)
            df_cleaned = df_cleaned[url_mask]
            logger.info(f"After URL cleaning: {len(df_cleaned)} rows")
            
            # 6. Drop unnecessary columns
            columns_to_drop_now = [col for col in ['Status Code', 'Status'] if col in df_cleaned.columns]
            if columns_to_drop_now:
                df_cleaned = df_cleaned.drop(columns_to_drop_now, axis=1)
            
            # 7. Create final DataFrame with standard columns
            final_df = pd.DataFrame()
            final_df['URL'] = df_cleaned[url_col]
            final_df['Embeddings'] = df_cleaned[embeddings_col]
            
            # 8. Remove duplicates
            initial_count = len(final_df)
            final_df = final_df.drop_duplicates(subset=['URL'], keep='first')
            duplicates_removed = initial_count - len(final_df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate URLs")
            
            # Update statistics
            self.processing_stats["embeddings_processed"] = original_size
            self.processing_stats["invalid_embeddings_removed"] = original_size - len(final_df)
            
            logger.info(f"Embeddings data cleaning completed. Final shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logger.error(f"Error cleaning embeddings data: {e}")
            return None
    
    def convert_embeddings_to_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert embedding strings to numpy arrays.
        
        Args:
            df: DataFrame with embeddings as strings
            
        Returns:
            DataFrame with embeddings as numpy arrays
        """
        logger.info("Converting embeddings to arrays")
        
        def parse_embedding(embedding_str: str) -> Optional[np.ndarray]:
            """Parse embedding string to numpy array."""
            try:
                # Clean the string
                cleaned = str(embedding_str).strip()
                
                # Remove brackets and quotes
                cleaned = cleaned.strip('[]').replace("'", "").replace('"', '')
                
                # Split and convert to floats
                values = [float(x.strip()) for x in cleaned.split(',')]
                
                return np.array(values)
                
            except Exception as e:
                logger.warning(f"Failed to parse embedding: {e}")
                return None
        
        df_result = df.copy()
        
        # Apply conversion
        df_result['Embeddings'] = df_result['Embeddings'].apply(parse_embedding)
        
        # Remove rows where conversion failed
        initial_count = len(df_result)
        df_result = df_result.dropna(subset=['Embeddings'])
        conversion_failures = initial_count - len(df_result)
        
        if conversion_failures > 0:
            logger.warning(f"Failed to convert {conversion_failures} embeddings")
        
        logger.info(f"Successfully converted {len(df_result)} embeddings to arrays")
        return df_result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "links_processed": 0,
            "embeddings_processed": 0,
            "invalid_urls_removed": 0,
            "invalid_embeddings_removed": 0
        }
        logger.info("Processing statistics reset")
