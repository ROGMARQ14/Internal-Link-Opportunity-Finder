"""
Similarity engine with caching for performance optimization.
"""

import numpy as np
import pandas as pd
import logging
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from config import CACHE_ENABLED, CACHE_TTL_SECONDS, CACHE_MAX_SIZE, CACHE_DIR

logger = logging.getLogger(__name__)

class SimilarityCache:
    """In-memory cache with TTL and size limits."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl

    def _evict_oldest(self):
        """Remove oldest entry when cache is full."""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
        del self.cache[oldest_key]
        logger.debug(f"Evicted cache entry: {oldest_key}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if self._is_expired(entry['timestamp']):
            del self.cache[key]
            return None

        return entry['value']

    def set(self, key: str, value: Any):
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl
        }

class SimilarityEngine:
    """Handles similarity calculations with caching and performance optimization."""

    def __init__(self, enable_cache: bool = CACHE_ENABLED):
        self.cache = SimilarityCache() if enable_cache else None
        self.enable_cache = enable_cache
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def _generate_cache_key(self, embeddings: np.ndarray, top_n: int) -> str:
        """Generate a unique cache key for embeddings and parameters."""
        # Create hash of embeddings array
        embeddings_hash = hashlib.md5(embeddings.tobytes()).hexdigest()
        return f"similarity_{embeddings_hash}_{top_n}"

    def _save_to_disk(self, key: str, data: Any):
        """Save data to disk cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': time.time()
                }, f)
            logger.debug(f"Saved to disk cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if not cache_file.exists():
                return None

            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Check if expired
            if time.time() - cached_data['timestamp'] > CACHE_TTL_SECONDS:
                cache_file.unlink()  # Remove expired file
                return None

            logger.debug(f"Loaded from disk cache: {key}")
            return cached_data['data']
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix with caching.

        Args:
            embeddings: Array of embeddings

        Returns:
            Cosine similarity matrix
        """
        if not self.enable_cache:
            return cosine_similarity(embeddings)

        # Generate cache key
        cache_key = self._generate_cache_key(embeddings, 0)  # 0 for matrix

        # Try in-memory cache first
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Using cached similarity matrix")
                return cached_result

        # Try disk cache
        cached_result = self._load_from_disk(cache_key)
        if cached_result is not None:
            # Store in memory cache for faster access
            if self.cache:
                self.cache.set(cache_key, cached_result)
            return cached_result

        # Calculate similarity matrix
        start_time = time.time()
        similarity_matrix = cosine_similarity(embeddings)
        calculation_time = time.time() - start_time

        logger.info(f"Calculated similarity matrix in {calculation_time:.2f} seconds")

        # Cache the result
        if self.cache:
            self.cache.set(cache_key, similarity_matrix)
        self._save_to_disk(cache_key, similarity_matrix)

        return similarity_matrix

    def find_related_pages(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Find top N related pages for each URL based on cosine similarity.

        Args:
            df: DataFrame with URL and Embeddings columns
            top_n: Number of related pages to find

        Returns:
            Dictionary mapping URLs to lists of related URLs
        """
        logger.info(f"Finding top {top_n} related pages for each URL...")

        # Extract embeddings and URLs
        embeddings = np.stack(df['Embeddings'].values)
        urls = df['URL'].values

        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings)

        # Find related pages
        related_pages = {}
        for idx, url in enumerate(urls):
            # Get similarity scores for this URL
            similarities = similarity_matrix[idx]

            # Find top N most similar URLs (excluding self)
            similar_indices = similarities.argsort()[-(top_n+1):][::-1]
            similar_indices = [i for i in similar_indices if urls[i] != url][:top_n]

            # Get the related URLs
            related_urls = urls[similar_indices].tolist()
            related_pages[url] = related_urls

        logger.info(f"Found related pages for {len(related_pages)} URLs")
        return related_pages

    def get_similarity_scores(self, df: pd.DataFrame, url1: str, url2: str) -> Optional[float]:
        """
        Get similarity score between two specific URLs.

        Args:
            df: DataFrame with URL and Embeddings columns
            url1: First URL
            url2: Second URL

        Returns:
            Similarity score or None if URLs not found
        """
        try:
            # Find indices of the URLs
            url_to_idx = {url: idx for idx, url in enumerate(df['URL'].values)}

            if url1 not in url_to_idx or url2 not in url_to_idx:
                return None

            idx1, idx2 = url_to_idx[url1], url_to_idx[url2]

            # Calculate similarity matrix if not cached
            embeddings = np.stack(df['Embeddings'].values)
            similarity_matrix = self.calculate_similarity_matrix(embeddings)

            return float(similarity_matrix[idx1][idx2])
        except Exception as e:
            logger.error(f"Error calculating similarity between {url1} and {url2}: {e}")
            return None

    def clear_cache(self):
        """Clear all caches."""
        if self.cache:
            self.cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info("All caches cleared")

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'enabled': self.enable_cache,
            'memory_cache': self.cache.stats() if self.cache else None,
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }
        return stats
