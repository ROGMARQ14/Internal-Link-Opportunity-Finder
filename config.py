"""
Configuration settings for the Internal Link Opportunity Finder.
"""
import os
from pathlib import Path

# --------------------------------------------------  App
APP_NAME        = "Internal Link Opportunity Finder"
APP_VERSION     = "2.0.0"
APP_DESCRIPTION = "Discover internal linking opportunities using vector embeddings"

# --------------------------------------------------  Files
MAX_FILE_SIZE_MB                = 1000
CHUNK_SIZE                      = 10_000
SAMPLE_SIZE_DELIMITER_DETECTION = 1024
LARGE_FILE_THRESHOLD_MB         = 50

# --------------------------------------------------  Analysis
DEFAULT_TOP_N = 10
MAX_TOP_N     = 20
MIN_TOP_N     = 1

# --------------------------------------------------  Cache
CACHE_ENABLED     = True
CACHE_TTL_SECONDS = 60 * 60   # 1 hour
CACHE_MAX_SIZE    = 1_000

# --------------------------------------------------  Cleaning rules
INVALID_URL_PATTERNS     = ['category/', 'tag/', 'sitemap', 'search', '/home/', 'index', '/page/']
INVALID_EMBEDDING_WORDS  = ['timeout', 'error', 'null', 'undefined', 'nan']
VALID_LINK_TYPES         = ['Hyperlink']
VALID_STATUS_CODES       = [200]
VALID_LINK_POSITIONS     = ['Content', 'Aside']
COLUMNS_TO_DROP          = ['Size (Bytes)', 'Follow', 'Target', 'Rel', 'Path Type', 'Link Path', 'Link Origin']
POTENTIAL_URL_COLUMNS    = ['URL', 'Address', 'Url', 'address']
CSV_DELIMITERS           = [';', ',', '\t', '|']

# --------------------------------------------------  Logging
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --------------------------------------------------  Paths
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
LOGS_DIR  = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
for _d in (DATA_DIR, LOGS_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)
