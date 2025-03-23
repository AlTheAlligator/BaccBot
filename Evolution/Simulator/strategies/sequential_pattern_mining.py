import logging
import numpy as np
from collections import defaultdict, Counter, deque

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SequentialPatternMiningStrategy(BaseStrategy):
    """
    A strategy that applies sequential pattern mining techniques to discover frequent patterns.
    
    This strategy identifies frequent sequential patterns in the outcome history and uses
    them to make predictions about future outcomes.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Sequential Pattern Mining strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.min_samples = params.get('min_samples', 20)  # Minimum samples before making predictions
        self.min_pattern_length = params.get('min_pattern_length', 2)  # Minimum pattern length to consider
        self.max_pattern_length = params.get('max_pattern_length', 6)  # Maximum pattern length to consider
        self.confidence_threshold = params.get('confidence_threshold', 0.65)  # Min confidence to place bet
        self.min_support = params.get('min_support', 3)  # Minimum occurrences for a pattern to be considered
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.use_weighted_patterns = params.get('use_weighted_patterns', True)  # Whether to weight patterns by length
        self.recency_factor = params.get('recency_factor', 0.2)  # How much to favor recent pattern occurrences
        self.pattern_timeout = params.get('pattern_timeout', 50)  # Outcome count before a pattern is considered "stale"
        self.use_confidence_scaling = params.get('use_confidence_scaling', True)  # Scale confidence by pattern quality
        
        # Pattern storage
        self.pattern_store = {}  # {pattern: {'P': count, 'B': count, 'last_seen': index}}
        self.pattern_predictions = deque(maxlen=100)  # Track prediction quality
        
        # Index tracking
        self.current_index = 0
        
        logger.info(f"Initialized Sequential Pattern Mining strategy with min_pattern_length={self.min_pattern_length}, "
                   f"max_pattern_length={self.max_pattern_length}, min_support={self.min_support}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on sequential pattern analysis.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable pattern mining
            logger.debug(f"Not enough data for pattern mining ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for pattern analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes for patterns ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update pattern database if we have new outcomes
        if self.current_index < len(non_tie_outcomes):
            self._update_patterns(non_tie_outcomes)
            self.current_index = len(non_tie_outcomes)
        
        # Get matching patterns for current state
        matching_patterns = self._find_matching_patterns(non_tie_outcomes)
        
        # If no patterns match or all are outdated, skip betting
        if not matching_patterns:
            logger.debug("No significant patterns matched current state")
            return "SKIP"
        
        # Calculate prediction probabilities based on matched patterns
        p_prob, b_prob, confidence, best_pattern = self._calculate_prediction_probs(matching_patterns)
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize probabilities
        total = p_prob + b_prob
        if total > 0:
            p_prob /= total
            b_prob /= total
        else:
            p_prob = b_prob = 0.5
        
        logger.debug(f"Pattern prediction: P={p_prob:.3f}, B={b_prob:.3f}, confidence={confidence:.3f}, "
                    f"best_pattern={best_pattern}")
        
        # Make decision based on probabilities and confidence
        if confidence >= self.confidence_threshold:
            if p_prob > b_prob:
                return "P"
            else:
                return "B"
        else:
            logger.debug(f"Confidence too low: {confidence:.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _update_patterns(self, outcomes):
        """
        Update the pattern database with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Process each possible pattern length
        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, len(outcomes))):
            # Process each possible pattern of this length
            for i in range(len(outcomes) - length):
                pattern = tuple(outcomes[i:i+length])
                
                # If pattern has a next outcome, record it
                if i + length < len(outcomes):
                    next_outcome = outcomes[i + length]
                    
                    # Initialize pattern entry if it doesn't exist
                    if pattern not in self.pattern_store:
                        self.pattern_store[pattern] = {'P': 0, 'B': 0, 'last_seen': 0}
                    
                    # Update counts
                    if next_outcome in ['P', 'B']:
                        self.pattern_store[pattern][next_outcome] += 1
                    
                    # Update last seen index
                    self.pattern_store[pattern]['last_seen'] = i + length
        
        # Prune old patterns if database gets too large
        if len(self.pattern_store) > 10000:
            self._prune_patterns(outcomes)
    
    def _prune_patterns(self, outcomes):
        """
        Remove old or infrequent patterns to keep database manageable.
        
        Args:
            outcomes: List of outcomes for reference
        """
        # Identify patterns to remove
        to_remove = []
        current_idx = len(outcomes) - 1
        
        for pattern, stats in self.pattern_store.items():
            total_occurrences = stats.get('P', 0) + stats.get('B', 0)
            
            # Remove if pattern is old or infrequent
            if (current_idx - stats['last_seen'] > self.pattern_timeout or 
                total_occurrences < self.min_support):
                to_remove.append(pattern)
        
        # Remove identified patterns
        for pattern in to_remove:
            del self.pattern_store[pattern]
            
        logger.debug(f"Pruned {len(to_remove)} patterns, {len(self.pattern_store)} remain")
    
    def _find_matching_patterns(self, outcomes):
        """
        Find patterns that match the current state.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            list: List of (pattern, stats) tuples for matching patterns
        """
        if not outcomes:
            return []
            
        matching_patterns = []
        
        # Check all pattern lengths from longest to shortest (prioritize more specific patterns)
        for length in range(min(self.max_pattern_length, len(outcomes)), self.min_pattern_length - 1, -1):
            # Get the suffix of current outcomes that matches this length
            current_pattern = tuple(outcomes[-length:])
            
            # Check if this pattern exists in our database
            if current_pattern in self.pattern_store:
                stats = self.pattern_store[current_pattern]
                total = stats.get('P', 0) + stats.get('B', 0)
                
                # Only consider patterns with sufficient support
                if total >= self.min_support:
                    matching_patterns.append((current_pattern, stats))
        
        return matching_patterns
    
    def _calculate_prediction_probs(self, matching_patterns):
        """
        Calculate prediction probabilities based on matching patterns.
        
        Args:
            matching_patterns: List of (pattern, stats) tuples
            
        Returns:
            tuple: (p_prob, b_prob, confidence, best_pattern)
        """
        if not matching_patterns:
            return 0.5, 0.5, 0, None
        
        weighted_p = 0
        weighted_b = 0
        total_weight = 0
        best_confidence = 0
        best_pattern = None
        
        for pattern, stats in matching_patterns:
            # Calculate pattern statistics
            p_count = stats.get('P', 0)
            b_count = stats.get('B', 0)
            total = p_count + b_count
            
            if total < self.min_support:
                continue
                
            # Calculate confidence for this pattern
            if p_count > b_count:
                pattern_confidence = p_count / total
                pattern_prediction = 'P'
            else:
                pattern_confidence = b_count / total
                pattern_prediction = 'B'
                
            # Track best individual pattern
            if pattern_confidence > best_confidence:
                best_confidence = pattern_confidence
                best_pattern = (pattern, pattern_prediction, pattern_confidence)
                
            # Calculate weight for this pattern
            if self.use_weighted_patterns:
                # Weight by length (longer patterns are more specific)
                length_weight = len(pattern) / self.max_pattern_length
                
                # Weight by confidence (more confident patterns get more weight)
                confidence_weight = pattern_confidence * 2 - 1  # Transform 0.5-1 to 0-1
                
                # Weight by recency (if pattern was recently observed)
                recency_weight = 1.0  # Default
                
                # Combine weights
                pattern_weight = length_weight * (1 + confidence_weight)
                
                if self.recency_factor > 0:
                    # Apply recency weight if configured
                    recency_weight = 1.0
                    pattern_weight *= recency_weight
            else:
                # Simple weighting - all patterns equal
                pattern_weight = 1.0
                
            # Add this pattern's contribution
            weighted_p += p_count / total * pattern_weight
            weighted_b += b_count / total * pattern_weight
            total_weight += pattern_weight
        
        # Normalize probabilities
        if total_weight > 0:
            p_prob = weighted_p / total_weight
            b_prob = weighted_b / total_weight
        else:
            p_prob = b_prob = 0.5
            
        # Calculate overall confidence
        confidence = max(p_prob, b_prob)
        
        # Apply confidence scaling if enabled
        if self.use_confidence_scaling and best_pattern:
            # Scale by quality of best pattern 
            scaling_factor = min(1.0, 0.7 + (best_confidence - 0.5) * 0.6)
            confidence *= scaling_factor
            
        return p_prob, b_prob, confidence, best_pattern
    
    def get_top_patterns(self, top_n=10):
        """Get the top predictive patterns for debugging."""
        if not self.pattern_store:
            return []
            
        # Calculate predictive power of each pattern
        pattern_scores = []
        
        for pattern, stats in self.pattern_store.items():
            p_count = stats.get('P', 0)
            b_count = stats.get('B', 0)
            total = p_count + b_count
            
            if total < self.min_support:
                continue
                
            # Calculate skew (how much it leans toward P or B)
            if total > 0:
                skew = abs(p_count - b_count) / total
            else:
                skew = 0
                
            pattern_scores.append((pattern, p_count, b_count, skew))
        
        # Sort by skew (patterns with clearer predictions first)
        pattern_scores.sort(key=lambda x: x[3], reverse=True)
        
        return pattern_scores[:top_n]