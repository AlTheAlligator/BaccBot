import logging
import numpy as np
from collections import defaultdict, Counter

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class FrequencyAnalysisStrategy(BaseStrategy):
    """
    A strategy that analyzes the frequency distributions of outcomes.
    
    This strategy looks for statistical patterns in outcome frequencies across
    different time windows and contexts to find betting opportunities.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Frequency Analysis strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.short_window = params.get('short_window', 5)  # Short analysis window
        self.medium_window = params.get('medium_window', 15)  # Medium analysis window
        self.long_window = params.get('long_window', 30)  # Long analysis window
        self.min_samples = params.get('min_samples', max(self.long_window, 20))  # Minimum samples before making predictions
        self.confidence_threshold = params.get('confidence_threshold', 0.58)  # Min confidence to place bet
        self.pattern_length = params.get('pattern_length', 3)  # Length of patterns for conditional frequency analysis
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.use_trend_adjustment = params.get('use_trend_adjustment', True)  # Whether to adjust for trends
        self.trend_weight = params.get('trend_weight', 0.3)  # Weight for trend adjustment factor
        self.use_pattern_adjustment = params.get('use_pattern_adjustment', True)  # Whether to adjust for pattern frequencies
        self.pattern_weight = params.get('pattern_weight', 0.4)  # Weight for pattern adjustment factor
        self.use_chi_square = params.get('use_chi_square', True)  # Whether to use chi-square test for significance
        self.significance_level = params.get('significance_level', 0.2)  # p-value threshold for significance
        self.clustering_method = params.get('clustering_method', 'multi_window')  # Method for frequency clustering
        
        # Initialize frequency tracking
        self.global_frequencies = {'P': 0, 'B': 0}
        self.window_frequencies = {
            'short': {'P': 0, 'B': 0},
            'medium': {'P': 0, 'B': 0},
            'long': {'P': 0, 'B': 0}
        }
        self.pattern_frequencies = defaultdict(lambda: {'P': 0, 'B': 0})
        
        # Store results of analysis for tracking/debugging
        self.analysis_results = []
        
        logger.info(f"Initialized Frequency Analysis strategy with windows={self.short_window},{self.medium_window},{self.long_window}, "
                   f"confidence_threshold={self.confidence_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on frequency analysis.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable frequency analysis
            logger.debug(f"Not enough data for frequency analysis ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update frequencies with all available data
        self._update_frequencies(non_tie_outcomes)
        
        # Analyze frequencies to get bet recommendation
        bet, confidence, analysis = self._analyze_frequencies(non_tie_outcomes)
        
        # Store analysis results for debugging
        self.analysis_results.append(analysis)
        
        # Make decision based on confidence
        if confidence >= self.confidence_threshold:
            return bet
        else:
            logger.debug(f"Confidence too low: {confidence:.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _update_frequencies(self, outcomes):
        """
        Update the frequency counts based on outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update global frequencies
        self.global_frequencies = Counter(outcomes)
        
        # Update window-based frequencies
        self.window_frequencies = {
            'short': Counter(outcomes[-self.short_window:]),
            'medium': Counter(outcomes[-self.medium_window:]),
            'long': Counter(outcomes[-self.long_window:])
        }
        
        # Update pattern-based frequencies if enabled
        if self.use_pattern_adjustment:
            self.pattern_frequencies.clear()
            
            # Process each pattern of specified length
            for i in range(len(outcomes) - self.pattern_length):
                pattern = tuple(outcomes[i:i+self.pattern_length])
                next_outcome = outcomes[i+self.pattern_length]
                
                # Update frequency counts for this pattern
                if next_outcome in ['P', 'B']:
                    self.pattern_frequencies[pattern][next_outcome] += 1
    
    def _analyze_frequencies(self, outcomes):
        """
        Analyze frequency patterns to determine the best bet.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (bet, confidence, analysis_dict)
        """
        # Calculate base probabilities from different windows
        probs = self._calculate_base_probabilities()
        
        # Apply trend adjustment if enabled
        if self.use_trend_adjustment:
            trend_factor = self._calculate_trend_factor(outcomes)
            probs['P'] += trend_factor * self.trend_weight
            probs['B'] -= trend_factor * self.trend_weight
            
            # Ensure probabilities stay in valid range
            probs['P'] = max(0.01, min(0.99, probs['P']))
            probs['B'] = max(0.01, min(0.99, probs['B']))
        
        # Apply pattern-based adjustment if enabled
        pattern_factor = 0
        if self.use_pattern_adjustment:
            pattern = tuple(outcomes[-self.pattern_length:])
            pattern_factor = self._calculate_pattern_factor(pattern)
            
            probs['P'] += pattern_factor * self.pattern_weight
            probs['B'] -= pattern_factor * self.pattern_weight
            
            # Ensure probabilities stay in valid range
            probs['P'] = max(0.01, min(0.99, probs['P']))
            probs['B'] = max(0.01, min(0.99, probs['B']))
        
        # Apply banker bias
        probs['B'] += self.banker_bias
        
        # Normalize to ensure they sum to 1
        total = probs['P'] + probs['B']
        probs['P'] /= total
        probs['B'] /= total
        
        # Statistical significance test
        is_significant = True
        p_value = 0.5  # Default no significance
        
        if self.use_chi_square:
            is_significant, p_value = self._check_significance(outcomes)
        
        # Determine the best bet
        bet = 'P' if probs['P'] > probs['B'] else 'B'
        
        # Calculate confidence (difference in probabilities)
        confidence = abs(probs['P'] - probs['B'])
        
        # Reduce confidence if distribution is not statistically significant
        if self.use_chi_square and not is_significant:
            confidence *= 0.8  # Reduce confidence by 20%
        
        # Create analysis dict for debugging
        analysis = {
            'probabilities': probs,
            'trend_factor': trend_factor if self.use_trend_adjustment else 0,
            'pattern_factor': pattern_factor if self.use_pattern_adjustment else 0,
            'statistical_significance': {
                'is_significant': is_significant,
                'p_value': p_value
            },
            'final_confidence': confidence
        }
        
        return bet, confidence, analysis
    
    def _calculate_base_probabilities(self):
        """
        Calculate base probabilities from window frequencies.
        
        Returns:
            dict: {'P': p_prob, 'B': b_prob}
        """
        # Calculate probability for each window
        window_probs = {}
        
        for window, counts in self.window_frequencies.items():
            total = counts.get('P', 0) + counts.get('B', 0)
            if total > 0:
                p_prob = counts.get('P', 0) / total
                b_prob = counts.get('B', 0) / total
                window_probs[window] = {'P': p_prob, 'B': b_prob}
            else:
                window_probs[window] = {'P': 0.5, 'B': 0.5}
        
        # Combine probabilities based on clustering method
        if self.clustering_method == 'multi_window':
            # Apply different weights to different windows (prioritize recent data)
            weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
            
            p_prob = sum(window_probs[w]['P'] * weights[w] for w in weights)
            b_prob = sum(window_probs[w]['B'] * weights[w] for w in weights)
        elif self.clustering_method == 'short_vs_long':
            # Compare short-term vs long-term frequencies
            # If they differ significantly, it suggests a trend change
            short_diff = window_probs['short']['P'] - window_probs['long']['P']
            
            if abs(short_diff) > 0.1:  # Significant difference threshold
                # Prioritize short-term if trends are changing
                p_prob = window_probs['short']['P'] * 0.7 + window_probs['long']['P'] * 0.3
                b_prob = window_probs['short']['B'] * 0.7 + window_probs['long']['B'] * 0.3
            else:
                # Use medium window for stability if no significant change
                p_prob = window_probs['medium']['P']
                b_prob = window_probs['medium']['B']
        else:
            # Default to medium window
            p_prob = window_probs['medium']['P']
            b_prob = window_probs['medium']['B']
        
        return {'P': p_prob, 'B': b_prob}
    
    def _calculate_trend_factor(self, outcomes):
        """
        Calculate a factor representing the current trend.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            float: Trend factor (-1 to 1, positive = trending towards P)
        """
        # Compare frequencies in different time windows
        short = self.window_frequencies['short']
        medium = self.window_frequencies['medium']
        
        short_total = short.get('P', 0) + short.get('B', 0)
        medium_total = medium.get('P', 0) + medium.get('B', 0)
        
        if short_total > 0 and medium_total > 0:
            short_p_ratio = short.get('P', 0) / short_total
            medium_p_ratio = medium.get('P', 0) / medium_total
            
            # Calculate trend factor
            trend_factor = short_p_ratio - medium_p_ratio
            
            # Scale to -1 to 1 range
            return max(-1, min(1, trend_factor * 2))
        
        return 0
    
    def _calculate_pattern_factor(self, pattern):
        """
        Calculate a factor based on pattern frequencies.
        
        Args:
            pattern: Tuple of recent outcomes
            
        Returns:
            float: Pattern factor (-1 to 1, positive = favoring P)
        """
        if pattern not in self.pattern_frequencies:
            return 0
            
        counts = self.pattern_frequencies[pattern]
        total = counts.get('P', 0) + counts.get('B', 0)
        
        if total < 3:  # Require at least a few observations
            return 0
            
        if total > 0:
            p_prob = counts.get('P', 0) / total
            b_prob = counts.get('B', 0) / total
            
            # Calculate factor, scaled to -1 to 1 range
            return (p_prob - b_prob) * 2
        
        return 0
    
    def _check_significance(self, outcomes):
        """
        Perform chi-square test to check for significant deviations from expected frequencies.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (is_significant, p_value)
        """
        # Use the short window for significance testing
        recent = [o for o in outcomes[-self.short_window:] if o in ['P', 'B']]
        
        if len(recent) < 5:  # Need reasonable sample size
            return False, 0.5
            
        # Count occurrences
        p_count = recent.count('P')
        b_count = recent.count('B')
        total = p_count + b_count
        
        # Expected counts (null hypothesis: equal probability)
        expected_p = total / 2
        expected_b = total / 2
        
        # Calculate chi-square statistic
        chi_square = ((p_count - expected_p) ** 2) / expected_p + ((b_count - expected_b) ** 2) / expected_b
        
        # Approximate p-value (1 degree of freedom)
        # For a proper implementation, use scipy.stats.chi2.sf(chi_square, 1)
        # This is a simplification
        if chi_square < 1.642:  # 20% significance level for 1 df
            p_value = 0.5
            is_significant = False
        elif chi_square < 2.706:  # 10% significance level
            p_value = 0.2
            is_significant = True
        elif chi_square < 3.841:  # 5% significance level
            p_value = 0.1
            is_significant = True
        else:
            p_value = 0.05
            is_significant = True
        
        return is_significant, p_value
    
    def get_frequencies(self):
        """Get current frequency statistics for debugging."""
        return {
            "global": dict(self.global_frequencies),
            "windows": {k: dict(v) for k, v in self.window_frequencies.items()},
            "patterns": {str(k): v for k, v in self.pattern_frequencies.items()},  # Convert keys to strings for JSON
            "recent_analysis": self.analysis_results[-5:] if self.analysis_results else []
        }