"""
CSV handling utilities with robust delimiter detection and error handling.
"""

import csv
import logging
import pandas as pd
import streamlit as st
from io import StringIO
from typing import Optional, List, Dict, Any
from config import (
    CHUNK_SIZE, SAMPLE_SIZE_DELIMITER_DETECTION, 
    LARGE_FILE_THRESHOLD_MB, CSV_DELIMITERS
)

logger = logging.getLogger(__name__)

class CSVHandler:
    """Handles CSV file reading with automatic delimiter detection and error handling."""

    def __init__(self):
        self.delimiters = CSV_DELIMITERS

    def detect_delimiter(self, file_content: bytes, sample_size: int = SAMPLE_SIZE_DELIMITER_DETECTION) -> str:
        """
        Detect the delimiter in a CSV file using multiple methods.

        Args:
            file_content: The file content as bytes
            sample_size: Size of sample to analyze

        Returns:
            Detected delimiter
        """
        # Convert bytes to string if needed
        if isinstance(file_content, bytes):
            try:
                file_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    file_content = file_content.decode('latin1')
                except:
                    file_content = file_content.decode('utf-8', errors='ignore')

        # Method 1: Try csv.Sniffer
        try:
            sample = file_content[:sample_size]
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=',;	|').delimiter
            logger.info(f"Detected delimiter using csv.Sniffer: '{delimiter}'")
            return delimiter
        except Exception as e:
            logger.warning(f"CSV Sniffer failed: {str(e)}")

        # Method 2: Manual detection by counting common delimiters
        sample = file_content[:sample_size]
        lines = sample.split('
')[:5]  # Check first 5 lines
        delimiter_counts = {}

        for delimiter in self.delimiters:
            counts = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    counts.append(line.count(delimiter))

            # Check if delimiter appears consistently across lines
            if counts and len(set(counts)) <= 2:  # Allow some variation
                delimiter_counts[delimiter] = sum(counts) / len(counts)

        if delimiter_counts:
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            logger.info(f"Detected delimiter using manual method: '{best_delimiter}'")
            return best_delimiter

        # Default fallback
        logger.warning("No delimiter detected, defaulting to comma")
        return ','

    def read_large_csv_safely(self, uploaded_file, chunk_size: int = CHUNK_SIZE) -> Optional[pd.DataFrame]:
        """
        Safely read large CSV files in chunks with error handling.

        Args:
            uploaded_file: Streamlit uploaded file object
            chunk_size: Size of chunks to read

        Returns:
            DataFrame or None if failed
        """
        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Read a sample to detect delimiter
            file_content = uploaded_file.read(10240)  # Read 10KB for detection
            uploaded_file.seek(0)  # Reset again

            detected_delimiter = self.detect_delimiter(file_content)

            # Read in chunks
            chunks = []
            for chunk in pd.read_csv(
                uploaded_file,
                chunksize=chunk_size,
                sep=detected_delimiter,
                on_bad_lines='skip',
                engine='python',
                encoding_errors='ignore'
            ):
                chunks.append(chunk)

            if not chunks:
                logger.error("No data could be read from the file")
                return None

            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"Successfully read large CSV with {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")

            # Try alternative methods
            try:
                uploaded_file.seek(0)
                result = pd.read_csv(uploaded_file, sep=None, on_bad_lines='skip', engine='python')
                logger.info("Successfully read CSV using alternative method")
                return result
            except Exception as e2:
                logger.error(f"Alternative reading method failed: {str(e2)}")
                return None

    def read_csv_with_auto_delimiter(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Read CSV file with automatic delimiter detection and error handling.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            DataFrame or None if failed
        """
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            file_size = len(file_content)
            uploaded_file.seek(0)  # Reset again

            logger.info(f"Processing file of size: {file_size / (1024*1024):.2f} MB")

            # For large files, use chunked reading
            if file_size > LARGE_FILE_THRESHOLD_MB * 1024 * 1024:
                logger.info("Large file detected. Using chunked reading...")
                return self.read_large_csv_safely(uploaded_file)

            # For smaller files, try multiple methods
            detected_delimiter = self.detect_delimiter(file_content)

            # Try multiple reading methods
            methods = [
                {"name": "Detected delimiter", "sep": detected_delimiter},
                {"name": "Pandas auto-detection", "sep": None},
                {"name": "Semicolon delimiter", "sep": ";"},
                {"name": "Comma delimiter", "sep": ","},
            ]

            for method in methods:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        sep=method["sep"],
                        on_bad_lines='skip',
                        engine='python'
                    )

                    logger.info(f"Successfully read CSV using {method['name']}")

                    # Verify the data looks reasonable
                    if len(df.columns) <= 1:
                        logger.warning(f"Only {len(df.columns)} column detected with {method['name']}")
                        continue

                    return df

                except Exception as e:
                    logger.warning(f"{method['name']} failed: {str(e)}")
                    continue

            # Last resort with very permissive settings
            try:
                uploaded_file.seek(0)
                logger.warning("Trying last resort parsing method...")
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
                logger.error(f"All CSV reading methods failed. Last error: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            return None
