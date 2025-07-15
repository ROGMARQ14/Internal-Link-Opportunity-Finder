"""
Debugging utilities for identifying and fixing false positives in link detection.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from src.utils.url_normalizer import url_normalizer

logger = logging.getLogger(__name__)


class LinkDebugger:
    """Helps identify and debug false positives in link detection."""
    
    def __init__(self, links_df: pd.DataFrame):
        self.links_df = links_df
        self.url_normalizer = url_normalizer
        
    def generate_debug_report(self, target_url: str, related_url: str) -> Dict[str, Any]:
        """
        Generate a detailed debug report for a specific URL pair.
        
        Args:
            target_url: The target URL
            related_url: The related URL
            
        Returns:
            Detailed debug information
        """
        report = {
            'target_url': target_url,
            'related_url': related_url,
            'normalized_target': self.url_normalizer.normalize(target_url),
            'normalized_related': self.url_normalizer.normalize(related_url),
            'exact_matches': [],
            'partial_matches': [],
            'similarity_analysis': {},
            'recommendations': []
        }
        
        # Find exact matches in links data
        exact_source_matches = self.links_df[
            (self.links_df['Source'] == target_url) & 
            (self.links_df['Destination'] == related_url)
        ]
        
        exact_dest_matches = self.links_df[
            (self.links_df['Source'] == related_url) & 
            (self.links_df['Destination'] == target_url)
        ]
        
        report['exact_matches'] = pd.concat([exact_source_matches, exact_dest_matches]).to_dict('records')
        
        # Find partial matches (normalized URLs)
        all_sources = self.links_df['Source'].tolist()
        all_dests = self.links_df['Destination'].tolist()
        
        # Check for similar URLs
        target_norm = report['normalized_target']
        related_norm = report['normalized_related']
        
        similar_sources = [
            s for s in all_sources 
            if target_norm and target_norm in self.url_normalizer.normalize(s)
        ]
        similar_dests = [
            d for d in all_dests 
            if related_norm and related_norm in self.url_normalizer.normalize(d)
        ]
        
        report['partial_matches'] = {
            'similar_sources': similar_sources[:5],  # Limit to first 5
            'similar_dests': similar_dests[:5]
        }
        
        # URL similarity analysis
        if target_norm and related_norm:
            target_parts = target_norm.split('/')
            related_parts = related_norm.split('/')
            
            # Find common path segments
            common_segments = []
            for i, (t_part, r_part) in enumerate(zip(target_parts, related_parts)):
                if t_part == r_part:
                    common_segments.append(t_part)
            
            report['similarity_analysis'] = {
                'common_path_segments': common_segments,
                'path_similarity_ratio': len(common_segments) / max(
                    len(target_parts), len(related_parts)
                ),
                'domain_match': (
                    target_norm.split('/')[2] == related_norm.split('/')[2] 
                    if len(target_norm.split('/')) > 2 else False
                )
            }
        
        # Generate recommendations
        if report['exact_matches']:
            report['recommendations'].append(
                "URLs are correctly linked - check if normalization is causing issues"
            )
        elif report['partial_matches']['similar_sources'] or report['partial_matches']['similar_dests']:
            report['recommendations'].append(
                "Similar URLs found - consider URL normalization issues"
            )
        else:
            report['recommendations'].append(
                "No matches found - URLs are genuinely not linked"
            )
            
        return report
    
    def find_suspicious_false_positives(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find potentially suspicious false positives in the analysis results.
        
        Args:
            analysis_df: The analysis DataFrame
            
        Returns:
            DataFrame with suspicious cases
        """
        suspicious_cases = []
        
        # Look for cases where URLs have very similar paths but are marked as "Not Found"
        for idx, row in analysis_df.iterrows():
            target_url = row['Target URL']
            
            # Check each related URL column
            for col in analysis_df.columns:
                if col.startswith('Related URL ') and pd.notna(row[col]):
                    related_url = row[col]
                    status_col = col.replace('Related URL ', 'URL ') + ' links to A?'
                    
                    if status_col in row and row[status_col] == "Not Found":
                        # Generate debug info for this pair
                        debug_info = self.generate_debug_report(target_url, related_url)
                        
                        # Check if URLs are very similar (potential false positive)
                        target_norm = debug_info['normalized_target']
                        related_norm = debug_info['normalized_related']
                        
                        if target_norm and related_norm:
                            # Simple similarity check
                            similarity = self._calculate_url_similarity(
                                target_norm, related_norm
                            )
                            
                            if similarity > 0.8:  # High similarity threshold
                                suspicious_cases.append({
                                    'target_url': target_url,
                                    'related_url': related_url,
                                    'similarity_score': similarity,
                                    'debug_info': debug_info,
                                    'row_index': idx
                                })
        
        return pd.DataFrame(suspicious_cases)
    
    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate similarity between two URLs."""
        if not url1 or not url2:
            return 0.0
            
        # Split into parts
        parts1 = url1.split('/')
        parts2 = url2.split('/')
        
        # Calculate Jaccard similarity
        set1 = set(parts1)
        set2 = set(parts2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def validate_link_detection(self, analysis_df: pd.DataFrame, sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate link detection accuracy with a sample.
        
        Args:
            analysis_df: The analysis DataFrame
            sample_size: Number of cases to validate
            
        Returns:
            Validation report
        """
        # Get random sample of "Not Found" cases
        not_found_cases = []
        
        for idx, row in analysis_df.iterrows():
            target_url = row['Target URL']
            
            for col in analysis_df.columns:
                if col.startswith('Related URL ') and pd.notna(row[col]):
                    related_url = row[col]
                    status_col = col.replace('Related URL ', 'URL ') + ' links to A?'
                    
                    if status_col in row and row[status_col] == "Not Found":
                        not_found_cases.append({
                            'target_url': target_url,
                            'related_url': related_url,
                            'row_index': idx
                        })
        
        # Take sample
        sample = not_found_cases[:sample_size]
        
        validation_results = {
            'total_not_found': len(not_found_cases),
            'sample_size': len(sample),
            'sample_results': []
        }
        
        # Validate each sample
        for case in sample:
            debug_info = self.generate_debug_report(
                case['target_url'], case['related_url']
            )
            
            # Manual verification flag (user would need to check these)
            case['debug_info'] = debug_info
            case['requires_manual_check'] = True
            
            validation_results['sample_results'].append(case)
        
        return validation_results
    
    def export_debug_report(self, analysis_df: pd.DataFrame, output_path: str = "debug_report.csv"):
        """
        Export a comprehensive debug report for manual review.
        
        Args:
            analysis_df: The analysis DataFrame
            output_path: Path to save the debug report
        """
        suspicious = self.find_suspicious_false_positives(analysis_df)
        
        if not suspicious.empty:
            # Add debug information to suspicious cases
            debug_data = []
            for _, row in suspicious.iterrows():
                debug_info = self.generate_debug_report(
                    row['target_url'], row['related_url']
                )
                
                debug_data.append({
                    'target_url': row['target_url'],
                    'related_url': row['related_url'],
                    'similarity_score': row['similarity_score'],
                    'normalized_target': debug_info['normalized_target'],
                    'normalized_related': debug_info['normalized_related'],
                    'exact_matches_found': len(debug_info['exact_matches']),
                    'recommendation': '; '.join(debug_info['recommendations'])
                })
            
            debug_df = pd.DataFrame(debug_data)
            debug_df.to_csv(output_path, index=False)
            logger.info(f"Debug report exported to {output_path}")
            return debug_df
        else:
            logger.info("No suspicious cases found for debugging")
            return pd.DataFrame()
