"""
CSV handling utilities with robust delimiter detection and error handling.
"""

import csv
import logging
import pandas as pd
from io import StringIO
from typing import Optional, List, Dict, Any
from pathlib import Path

from config import (
    MAX_FILE_SIZE_MB, CHUNK_SIZE, SAMPLE_SIZE_DELIMITER_DETECTION,
    LARGE_FILE_THRESHOLD_MB, CSV_DELIMITERS, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class CSVHandler:
    """Production-grade CSV handler with automatic delimiter detection and error handling."""
    
    def __init__(self):
        self.supported_delimiters = CSV_DELIMITERS
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.chunk_size = CHUNK_SIZE
        self.sample_size = SAMPLE_SIZE_DELIMITER_DETECTION
        self.large_file_threshold = LARGE_FILE_THRESHOLD_MB * 1024 * 1024
        
    def detect_delimiter(self, file_content: str) -> str:
        """
        Detect CSV delimiter using multiple methods.
        
        Args:
            file_content: String content of the CSV file
            
        Returns:
            Detected delimiter character
        """
        try:
            # Method 1: CSV Sniffer
            sample = file_content[:self.sample_size]
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=''.join(self.supported_delimiters)).delimiter
            logger.info(f"Detected delimiter using CSV Sniffer: '{delimiter}'")
            return delimiter
            
        except Exception as e:
            logger.warning(f"CSV Sniffer failed: {e}")
            
        # Method 2: Manual detection by counting
        sample = file_content[:self.sample_size]
        lines = [line for line in sample.split('\n')[:5] if line.strip()]
        
        delimiter_scores = {}
        for delimiter in self.supported_delimiters:
            counts = [line.count(delimiter) for line in lines]
            if counts and len(set(counts)) <= 2:  # Consistent counts
                delimiter_scores[delimiter] = sum(counts) / len(counts)
                
        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            logger.info(f"Detected delimiter using manual method: '{best_delimiter}'")
            return best_delimiter
            
        # Fallback
        logger.warning("No delimiter detected, using comma as default")
        return ','
    
    def read_csv_chunked(self, file_path: Path, delimiter: str) -> Optional[pd.DataFrame]:
        """
        Read large CSV files in chunks.
        
        Args:
            file_path: Path to the CSV file
            delimiter: Delimiter to use
            
        Returns:
            Combined DataFrame or None if failed
        """
        try:
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                sep=delimiter,
                on_bad_lines='skip',
                engine='python',
                encoding_errors='ignore'
            ):
                chunks.append(chunk)
                chunk_count += 1
                logger.debug(f"Processed chunk {chunk_count}, rows: {len(chunk)}")
                
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully read {len(result)} rows from {chunk_count} chunks")
                return result
            else:
                logger.error("No chunks were successfully read")
                return None
                
        except Exception as e:
            logger.error(f"Chunked reading failed: {e}")
            return None
    
    def read_csv_with_auto_delimiter(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Read CSV with automatic delimiter detection and error handling.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame or None if failed
        """
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read content for analysis
            if hasattr(uploaded_file, 'read'):
                file_content = uploaded_file.read()
                file_size = len(file_content)
                uploaded_file.seek(0)  # Reset again
            else:
                logger.error("Invalid file object")
                return None
                
            # Check file size
            if file_size > self.max_file_size:
                logger.error(f"File too large: {file_size / (1024*1024):.2f} MB > {MAX_FILE_SIZE_MB} MB")
                return None
                
            logger.info(f"Processing file of size: {file_size / (1024*1024):.2f} MB")
            
            # Convert bytes to string if needed
            if isinstance(file_content, bytes):
                try:
                    file_content_str = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        file_content_str = file_content.decode('latin1')
                    except UnicodeDecodeError:
                        file_content_str = file_content.decode('utf-8', errors='ignore')
                        logger.warning("Used error-ignore decoding due to encoding issues")
            else:
                file_content_str = file_content
                
            # Detect delimiter
            delimiter = self.detect_delimiter(file_content_str)
            
            # For large files, use chunked reading
            if file_size > self.large_file_threshold:
                logger.info("Large file detected, using chunked reading")
                # Save to temp file for chunked reading
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                    tmp.write(file_content_str)
                    tmp_path = Path(tmp.name)
                
                try:
                    result = self.read_csv_chunked(tmp_path, delimiter)
                    return result
                finally:
                    tmp_path.unlink()  # Clean up temp file
            
            # For smaller files, try multiple methods
            methods = [
                {
                    "name": "Detected delimiter",
                    "params": {"sep": delimiter, "on_bad_lines": 'skip', "engine": 'python'}
                },
                {
                    "name": "Pandas auto-detection", 
                    "params": {"sep": None, "on_bad_lines": 'skip', "engine": 'python'}
                },
                {
                    "name": "Semicolon fallback",
                    "params": {"sep": ";", "on_bad_lines": 'skip', "engine": 'python'}
                },
                {
                    "name": "Comma fallback",
                    "params": {"sep": ",", "on_bad_lines": 'skip', "engine": 'python'}
                }
            ]
            
            for method in methods:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, **method["params"])
                    
                    # Validate result
                    if len(df.columns) <= 1:
                        logger.warning(f"{method['name']}: Only {len(df.columns)} column(s) detected")
                        continue
                        
                    logger.info(f"Successfully read CSV using {method['name']}")
                    logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                    return df
                    
                except Exception as e:
                    logger.warning(f"{method['name']} failed: {e}")
                    continue
            
            # Last resort with very permissive settings
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine='python',
                    on_bad_lines='skip',
                    encoding_errors='ignore',
                    quoting=csv.QUOTE_NONE
                )
                logger.info("Successfully read CSV using last resort method")
                return df
                
            except Exception as e:
                logger.error(f"All reading methods failed. Last error: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Critical error in CSV reading: {e}")
            return None
    
    def validate_csv_structure(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate CSV structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
        }
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            validation_result["valid"] = False
            validation_result["errors"].append("DataFrame is empty")
        
        # Check for excessive null values
        null_percentages = (df.isnull().sum() / len(df)) * 100
        high_null_columns = null_percentages[null_percentages > 50].index.tolist()
        if high_null_columns:
            validation_result["warnings"].append(f"High null percentage in columns: {high_null_columns}")
        
        logger.info(f"CSV validation completed. Valid: {validation_result['valid']}")
        return validation_result
