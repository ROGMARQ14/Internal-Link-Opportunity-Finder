"""
URL normalization utilities to ensure consistent URL matching across datasets.
"""

import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class URLNormalizer:
    """Handles URL normalization for consistent matching across datasets."""
    
    def __init__(self):
        self.strip_www = True
        self.strip_trailing_slash = True
        self.force_https = True
        self.strip_utm_params = True
        self.strip_fragment = True
    
    def normalize(self, url: str) -> Optional[str]:
        """
        Normalize a URL for consistent comparison.
        
        Args:
            url: Raw URL string
            
        Returns:
            Normalized URL or None if invalid
        """
        if not url or not isinstance(url, str):
            return None
            
        try:
            # Basic cleanup
            url = url.strip()
            
            # Handle relative URLs
            if url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                # Relative URLs can't be normalized without base domain
                return None
                
            # Parse URL
            parsed = urlparse(url)
            
            # Skip if no netloc
            if not parsed.netloc:
                return None
                
            # Normalize scheme
            scheme = 'https' if self.force_https else parsed.scheme.lower()
            
            # Normalize netloc
            netloc = parsed.netloc.lower()
            
            # Remove www if specified
            if self.strip_www and netloc.startswith('www.'):
                netloc = netloc[4:]
                
            # Handle common variations
            netloc = netloc.replace('//', '/')
            
            # Normalize path
            path = parsed.path
            if self.strip_trailing_slash and path.endswith('/') and len(path) > 1:
                path = path.rstrip('/')
                
            # Normalize path segments
            path = re.sub(r'/+', '/', path)
            
            # Handle query parameters
            query = parsed.query
            if self.strip_utm_params and query:
                query_params = parse_qs(query)
                # Remove UTM and tracking parameters
                tracking_params = {
                    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 
                    'utm_content', 'gclid', 'fbclid', 'ref', 'source'
                }
                filtered_params = {
                    k: v for k, v in query_params.items() 
                    if k.lower() not in tracking_params
                }
                query = urlencode(filtered_params, doseq=True)
                
            # Remove fragment
            fragment = '' if self.strip_fragment else parsed.fragment
            
            # Reconstruct URL
            normalized = urlunparse((
                scheme,
                netloc,
                path,
                parsed.params,
                query,
                fragment
            ))
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Failed to normalize URL '{url}': {e}")
            return None
    
    def normalize_batch(self, urls: list) -> list:
        """Normalize a batch of URLs."""
        return [self.normalize(url) for url in urls if url]
    
    def create_url_mapping(self, urls: list) -> dict:
        """
        Create a mapping from original URLs to normalized URLs.
        
        Args:
            urls: List of original URLs
            
        Returns:
            Dictionary mapping original -> normalized URLs
        """
        mapping = {}
        for url in urls:
            normalized = self.normalize(url)
            if normalized:
                mapping[url] = normalized
        return mapping
    
    def find_duplicate_groups(self, urls: list) -> dict:
        """
        Find groups of URLs that normalize to the same value.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            Dictionary mapping normalized URL -> list of original URLs
        """
        groups = {}
        for url in urls:
            normalized = self.normalize(url)
            if normalized:
                if normalized not in groups:
                    groups[normalized] = []
                groups[normalized].append(url)
        return groups


# Global normalizer instance
url_normalizer = URLNormalizer()
