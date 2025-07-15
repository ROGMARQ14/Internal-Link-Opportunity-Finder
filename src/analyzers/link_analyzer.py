"""
Link analyzer with bidirectional checking and quality assessment.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from config import DEFAULT_TOP_N
from src.utils.url_normalizer import url_normalizer

logger = logging.getLogger(__name__)

class LinkQualityAssessment:
    """Assess the quality of existing links."""

    def __init__(self):
        self.quality_metrics = [
            'anchor_text_relevance',
            'link_count',
            'context_quality',
            'link_authority'
        ]

    def assess_anchor_text_quality(self, anchor_texts: List[str]) -> Dict[str, Any]:
        """
        Assess the quality of anchor texts.

        Args:
            anchor_texts: List of anchor texts

        Returns:
            Quality assessment metrics
        """
        if not anchor_texts:
            return {
                'quality_score': 0.0,
                'descriptive_ratio': 0.0,
                'keyword_rich_ratio': 0.0,
                'generic_ratio': 0.0,
                'total_anchors': 0
            }

        # Generic anchor text patterns
        generic_patterns = [
            'click here', 'read more', 'learn more', 'here', 'this',
            'link', 'url', 'website', 'page', 'more info'
        ]

        # Analyze anchor texts
        total_anchors = len(anchor_texts)
        descriptive_count = 0
        keyword_rich_count = 0
        generic_count = 0

        for anchor in anchor_texts:
            if not anchor or pd.isna(anchor):
                continue

            anchor_lower = str(anchor).lower().strip()

            # Check if generic
            if any(pattern in anchor_lower for pattern in generic_patterns):
                generic_count += 1
            # Check if descriptive (more than 3 words)
            elif len(anchor_lower.split()) > 3:
                descriptive_count += 1
            # Check if keyword-rich (contains meaningful keywords)
            elif len(anchor_lower.split()) > 1:
                keyword_rich_count += 1

        # Calculate ratios
        descriptive_ratio = descriptive_count / total_anchors if total_anchors > 0 else 0
        keyword_rich_ratio = keyword_rich_count / total_anchors if total_anchors > 0 else 0
        generic_ratio = generic_count / total_anchors if total_anchors > 0 else 0

        # Calculate overall quality score
        quality_score = (descriptive_ratio * 1.0 + keyword_rich_ratio * 0.7 - generic_ratio * 0.5)
        quality_score = max(0, min(1, quality_score))  # Clamp to [0, 1]

        return {
            'quality_score': quality_score,
            'descriptive_ratio': descriptive_ratio,
            'keyword_rich_ratio': keyword_rich_ratio,
            'generic_ratio': generic_ratio,
            'total_anchors': total_anchors,
            'anchor_examples': anchor_texts[:5]  # First 5 examples
        }

    def assess_link_context(self, source_url: str, dest_url: str) -> Dict[str, Any]:
        """
        Assess the contextual relevance of a link.

        Args:
            source_url: Source URL
            dest_url: Destination URL

        Returns:
            Context quality metrics
        """
        # Extract path components for analysis
        try:
            source_path = source_url.split('/')[-1].replace('-', ' ').replace('_', ' ')
            dest_path = dest_url.split('/')[-1].replace('-', ' ').replace('_', ' ')

            # Simple keyword overlap analysis
            source_keywords = set(source_path.lower().split())
            dest_keywords = set(dest_path.lower().split())

            # Calculate keyword overlap
            overlap = len(source_keywords.intersection(dest_keywords))
            total_keywords = len(source_keywords.union(dest_keywords))

            overlap_ratio = overlap / total_keywords if total_keywords > 0 else 0

            return {
                'keyword_overlap_ratio': overlap_ratio,
                'shared_keywords': list(source_keywords.intersection(dest_keywords)),
                'context_score': overlap_ratio  # Simple context score
            }
        except Exception as e:
            logger.warning(f"Error assessing context for {source_url} -> {dest_url}: {e}")
            return {
                'keyword_overlap_ratio': 0.0,
                'shared_keywords': [],
                'context_score': 0.0
            }

class LinkAnalyzer:
    """Enhanced link analyzer with bidirectional checking and quality assessment."""

    def __init__(self, links_df: pd.DataFrame):
        """
        Initialize the analyzer with links data.

        Args:
            links_df: DataFrame with 'Source' and 'Destination' columns
        """
        self.links_df = links_df
        self.quality_assessor = LinkQualityAssessment()
        self._validate_links_data()
        self._create_link_indices()
        self._create_normalized_indices()

    def _validate_links_data(self):
        """Validate the links DataFrame structure."""
        required_columns = ['Source', 'Destination']
        missing_columns = [col for col in required_columns if col not in self.links_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Initialized LinkAnalyzer with {len(self.links_df)} links")

    def _create_link_indices(self):
        """Create indices for faster lookups."""
        self.source_to_destinations = defaultdict(set)
        self.destination_to_sources = defaultdict(set)

        for _, row in self.links_df.iterrows():
            source = row['Source']
            destination = row['Destination']

            self.source_to_destinations[source].add(destination)
            self.destination_to_sources[destination].add(source)

        logger.info("Created link indices for faster lookups")

    def _create_normalized_indices(self):
        """Create normalized URL indices for better matching."""
        self.normalized_source_to_destinations = defaultdict(set)
        self.normalized_destination_to_sources = defaultdict(set)
        self.original_to_normalized = {}
        self.normalized_to_originals = defaultdict(list)
        
        # Build normalized indices
        for _, row in self.links_df.iterrows():
            original_source = row['Source']
            original_dest = row['Destination']
            
            # Normalize URLs
            norm_source = url_normalizer.normalize(original_source)
            norm_dest = url_normalizer.normalize(original_dest)
            
            if norm_source and norm_dest:
                # Store mappings
                self.original_to_normalized[original_source] = norm_source
                self.original_to_normalized[original_dest] = norm_dest
                
                self.normalized_to_originals[norm_source].append(original_source)
                self.normalized_to_originals[norm_dest].append(original_dest)
                
                # Build normalized indices
                self.normalized_source_to_destinations[norm_source].add(norm_dest)
                self.normalized_destination_to_sources[norm_dest].add(norm_source)
        
        logger.info(f"Created normalized indices with {len(self.normalized_to_originals)} unique normalized URLs")

    def check_link_exists_bidirectional(self, url1: str, url2: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if there's a link between two URLs in either direction.

        Args:
            url1: First URL (typically target URL)
            url2: Second URL (typically related URL)

        Returns:
            tuple: (exists, status, metadata)
        """
        try:
            # Normalize URLs for consistent matching
            norm_url1 = url_normalizer.normalize(url1)
            norm_url2 = url_normalizer.normalize(url2)
            
            if not norm_url1 or not norm_url2:
                return False, "Invalid URL", {'error': 'URL normalization failed'}
            
            # Check both directions using normalized indices
            direction1_exists = norm_url1 in self.normalized_source_to_destinations.get(norm_url2, set())
            direction2_exists = norm_url2 in self.normalized_source_to_destinations.get(norm_url1, set())
            
            # Also check original indices as fallback
            orig_direction1 = url1 in self.source_to_destinations.get(url2, set())
            orig_direction2 = url2 in self.source_to_destinations.get(url1, set())
            
            # Use normalized results, but log discrepancies
            exists = direction1_exists or direction2_exists
            original_exists = orig_direction1 or orig_direction2
            
            if exists != original_exists:
                logger.debug(f"URL normalization affected result: {url1} <-> {url2}")
                logger.debug(f"Normalized: {norm_url1} <-> {norm_url2}")
                logger.debug(f"Normalized result: {exists}, Original result: {original_exists}")

            # Gather metadata
            metadata = {
                'url1': url1,
                'url2': url2,
                'normalized_url1': norm_url1,
                'normalized_url2': norm_url2,
                'direction1_exists': direction1_exists,  # url2 -> url1
                'direction2_exists': direction2_exists,  # url1 -> url2
                'original_direction1_exists': orig_direction1,
                'original_direction2_exists': orig_direction2,
                'total_outlinks_url1': len(self.normalized_source_to_destinations.get(norm_url1, set())),
                'total_outlinks_url2': len(self.normalized_source_to_destinations.get(norm_url2, set())),
                'total_inlinks_url1': len(self.normalized_destination_to_sources.get(norm_url1, set())),
                'total_inlinks_url2': len(self.normalized_destination_to_sources.get(norm_url2, set()))
            }

            # Determine status
            if direction1_exists and direction2_exists:
                return True, "Exists (Bidirectional)", metadata
            elif direction1_exists:
                return True, "Exists (Related → Target)", metadata
            elif direction2_exists:
                return True, "Exists (Target → Related)", metadata
            else:
                return False, "Not Found", metadata

        except Exception as e:
            logger.error(f"Error checking link between {url1} and {url2}: {str(e)}")
            return False, "Error", {'error': str(e)}

    def assess_link_quality(self, source_url: str, dest_url: str) -> Dict[str, Any]:
        """
        Assess the quality of existing links between URLs.

        Args:
            source_url: Source URL
            dest_url: Destination URL

        Returns:
            Quality assessment metrics
        """
        # Get all links between the URLs
        links_forward = self.links_df[
            (self.links_df['Source'] == source_url) & 
            (self.links_df['Destination'] == dest_url)
        ]

        links_backward = self.links_df[
            (self.links_df['Source'] == dest_url) & 
            (self.links_df['Destination'] == source_url)
        ]

        all_links = pd.concat([links_forward, links_backward])

        if len(all_links) == 0:
            return {
                'quality_score': 0.0,
                'link_count': 0,
                'anchor_assessment': self.quality_assessor.assess_anchor_text_quality([]),
                'context_assessment': self.quality_assessor.assess_link_context(source_url, dest_url)
            }

        # Extract anchor texts
        anchor_texts = []
        if 'Anchor' in all_links.columns:
            anchor_texts = all_links['Anchor'].dropna().tolist()

        # Assess anchor text quality
        anchor_assessment = self.quality_assessor.assess_anchor_text_quality(anchor_texts)

        # Assess context quality
        context_assessment = self.quality_assessor.assess_link_context(source_url, dest_url)

        # Calculate overall quality score
        anchor_weight = 0.6
        context_weight = 0.4

        overall_quality = (
            anchor_assessment['quality_score'] * anchor_weight +
            context_assessment['context_score'] * context_weight
        )

        return {
            'quality_score': overall_quality,
            'link_count': len(all_links),
            'anchor_assessment': anchor_assessment,
            'context_assessment': context_assessment,
            'link_details': all_links.to_dict('records')
        }

    def calculate_opportunity_score(self, exists: bool, metadata: Dict[str, Any], 
                                  quality_assessment: Dict[str, Any]) -> float:
        """
        Calculate an opportunity score for prioritizing linking opportunities.

        Args:
            exists: Whether link already exists
            metadata: Link metadata
            quality_assessment: Quality assessment results

        Returns:
            Opportunity score (0-1, higher = better opportunity)
        """
        if exists:
            # If link exists, score based on quality improvement potential
            current_quality = quality_assessment.get('quality_score', 0.0)
            return max(0, 1 - current_quality)  # Room for improvement

        # Score based on authority and linking patterns
        url1_authority = min(metadata.get('total_inlinks_url1', 0), 50) / 50.0
        url2_authority = min(metadata.get('total_inlinks_url2', 0), 50) / 50.0

        # Higher score for linking between authoritative pages
        authority_score = (url1_authority + url2_authority) / 2.0

        # Context relevance from quality assessment
        context_score = quality_assessment.get('context_assessment', {}).get('context_score', 0.0)

        # Combine factors
        opportunity_score = (authority_score * 0.4 + context_score * 0.6)

        return min(1.0, opportunity_score)

    def analyze_opportunities(self, related_pages: Dict[str, List[str]], 
                            top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
        """
        Analyze internal linking opportunities with improved accuracy.

        Args:
            related_pages: Dictionary mapping target URLs to lists of related URLs
            top_n: Number of related URLs to analyze per target

        Returns:
            DataFrame with analysis results
        """
        logger.info(f"Analyzing opportunities for {len(related_pages)} target URLs")

        output_data = []

        for target_url, related_urls in related_pages.items():
            # Get existing inlinks for this URL
            inlinks = list(self.destination_to_sources.get(target_url, set()))
            links_to_target = ', '.join(inlinks) if inlinks else "none"

            # Prepare row data
            row = {
                'Target URL': target_url,
                'Links to Target URL': links_to_target,
                'Total Inlinks': len(inlinks)
            }

            # Pad related URLs to ensure consistent structure
            padded_related_urls = related_urls + [None] * (top_n - len(related_urls))

            # Analyze each related URL
            for i, related_url in enumerate(padded_related_urls, 1):
                if related_url is not None:
                    # Check if link exists (bidirectional)
                    exists, status, metadata = self.check_link_exists_bidirectional(
                        target_url, related_url
                    )

                    # Assess link quality
                    quality_assessment = self.assess_link_quality(target_url, related_url)

                    # Calculate opportunity score
                    opportunity_score = self.calculate_opportunity_score(
                        exists, metadata, quality_assessment
                    )

                    # Add to row
                    row[f'Related URL {i}'] = related_url
                    row[f'URL {i} links to A?'] = "Exists" if exists else "Not Found"
                    row[f'URL {i} Status Details'] = status
                    row[f'URL {i} Quality Score'] = quality_assessment['quality_score']
                    row[f'URL {i} Opportunity Score'] = opportunity_score
                    row[f'URL {i} Link Count'] = quality_assessment['link_count']

                    # Add context information
                    if quality_assessment['context_assessment']:
                        row[f'URL {i} Context Score'] = quality_assessment['context_assessment']['context_score']

                else:
                    # No related URL
                    row[f'Related URL {i}'] = None
                    row[f'URL {i} links to A?'] = "Not Found"
                    row[f'URL {i} Status Details'] = "No related URL"
                    row[f'URL {i} Quality Score'] = 0.0
                    row[f'URL {i} Opportunity Score'] = 0.0
                    row[f'URL {i} Link Count'] = 0
                    row[f'URL {i} Context Score'] = 0.0

            output_data.append(row)

        result_df = pd.DataFrame(output_data)
        logger.info(f"Completed analysis for {len(result_df)} target URLs")
        return result_df

    def get_link_statistics(self) -> Dict[str, Any]:
        """Get comprehensive link statistics."""
        return {
            'total_links': len(self.links_df),
            'unique_sources': len(self.source_to_destinations),
            'unique_destinations': len(self.destination_to_sources),
            'avg_outlinks_per_source': len(self.links_df) / len(self.source_to_destinations) if self.source_to_destinations else 0,
            'avg_inlinks_per_destination': len(self.links_df) / len(self.destination_to_sources) if self.destination_to_sources else 0,
            'top_linking_sources': self._get_top_sources(),
            'top_linked_destinations': self._get_top_destinations()
        }

    def _get_top_sources(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get top sources by outlink count."""
        source_counts = [(source, len(destinations)) 
                        for source, destinations in self.source_to_destinations.items()]
        return sorted(source_counts, key=lambda x: x[1], reverse=True)[:top_n]

    def _get_top_destinations(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get top destinations by inlink count."""
        dest_counts = [(dest, len(sources)) 
                      for dest, sources in self.destination_to_sources.items()]
        return sorted(dest_counts, key=lambda x: x[1], reverse=True)[:top_n]
