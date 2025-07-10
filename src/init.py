"""
Internal Link Opportunity Finder 2.0

A production-grade application for discovering internal linking opportunities 
using vector embeddings and semantic similarity analysis.
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Core imports for external use
from .core.similarity_engine import SimilarityEngine
from .core.data_processor import DataProcessor
from .analyzers.link_analyzer import LinkAnalyzer
from .utils.csv_handler import CSVHandler
from .utils.export_utils import ExportManager

__all__ = [
    "SimilarityEngine",
    "DataProcessor", 
    "LinkAnalyzer",
    "CSVHandler",
    "ExportManager"
]
