import logging
import numpy as np
from collections import defaultdict, Counter

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class PatternInterruptionStrategy(BaseStrategy):
    """
    A strategy that identifies when established patterns are interrupted.
    
    This strategy detects when recurring patterns in the game outcomes are suddenly broken,
    which may indicate a shift in the game's dynamics and a potential betting opportunity.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Pattern Interruption strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.pattern_window = params.get('pattern_window', 8)  # Window to look for patterns
        self.min_samples = params.get('min_samples', 15)  # Minimum samples before making predictions
        self.repeat_threshold = params.get('repeat_threshold', 2)  # Min repeats to establish a pattern
        self.confidence_threshold = params.get('confidence_threshold', 0.6)  # Min confidence to place bet
        self.post_interruption_followup = params.get('post_interruption_followup', True)  # Whether to bet on follow-up hands
        self.max_followup_hands = params.get('max_followup_hands', 2)  # Max hands to follow up after interruption
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        
        # Pattern tracking
        self.pattern_store = defaultdict(Counter)  # Store observed pattern continuations
        self.recent_interruptions = []  # Track recent pattern interruptions for follow-up bets
        self.active_followups = 0  # Counter for active follow-up bets
        
        logger.info(f"Initialized Pattern Interruption strategy with pattern_window={self.pattern_window}, "
                   f"confidence_threshold={self.confidence_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on pattern interruptions.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data to identify patterns reliably
            logger.debug(f"Not enough data for pattern interruption analysis ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for pattern analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update pattern store with recent outcomes
        self._update_pattern_store(non_tie_outcomes)
        
        # First, check if we're in follow-up mode from a previous interruption
        if self.post_interruption_followup and self.active_followups > 0:
            followup_bet = self._get_followup_bet()
            self.active_followups -= 1
            
            if followup_bet in ['P', 'B']:
                logger.debug(f"Making follow-up bet: {followup_bet} (remaining: {self.active_followups})")
                return followup_bet
        
        # Look for a pattern interruption in the most recent outcomes
        interruption = self._detect_pattern_interruption(non_tie_outcomes)
        if interruption:
            pattern, expected, actual, confidence = interruption
            
            logger.debug(f"Detected pattern interruption: {pattern} -> expected {expected}, got {actual}, confidence: {confidence:.3f}")
            
            # If the interruption is significant enough, use it for betting
            if confidence >= self.confidence_threshold:
                # Reset and start follow-up bets
                self.recent_interruptions.append((pattern, actual))
                self.active_followups = self.max_followup_hands
                
                # Return the interrupting side as our bet (betting on continuation of the new pattern)
                return actual
            else:
                logger.debug(f"Interruption confidence too low: {confidence:.3f} < {self.confidence_threshold}")
        
        return "SKIP"
    
    def _update_pattern_store(self, outcomes):
        """
        Update the pattern store with observed transitions.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update pattern continuations for various pattern lengths
        for pattern_length in range(2, min(6, len(outcomes) // 3)):
            # Process each pattern of this length
            for i in range(len(outcomes) - pattern_length - 1):
                pattern = tuple(outcomes[i:i+pattern_length])
                next_outcome = outcomes[i+pattern_length]
                
                # Update the continuation counter for this pattern
                if next_outcome in ['P', 'B']:
                    self.pattern_store[pattern][next_outcome] += 1
    
    def _detect_pattern_interruption(self, outcomes):
        """
        Detect if a pattern was interrupted in the recent outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple or None: (pattern, expected, actual, confidence) if interruption detected, None otherwise
        """
        # Only look at the most recent outcomes
        recent = outcomes[-self.pattern_window:]
        
        # Look for patterns of different lengths
        for pattern_length in range(2, min(5, len(recent) - 1)):
            # Check for repeated patterns in the recent outcomes
            for start_idx in range(len(recent) - pattern_length):
                pattern = tuple(recent[start_idx:start_idx+pattern_length])
                
                # Check if this pattern appears multiple times
                appearances = 0
                for i in range(len(recent) - pattern_length):
                    if tuple(recent[i:i+pattern_length]) == pattern:
                        appearances += 1
                
                # If the pattern appears enough times to establish expectation
                if appearances >= self.repeat_threshold:
                    # Check if the pattern was recently interrupted
                    # Look at the last occurrence of the pattern
                    for i in range(len(recent) - pattern_length, 0, -1):
                        if i + pattern_length < len(recent) and tuple(recent[i:i+pattern_length]) == pattern:
                            # Found the last occurrence, check what followed
                            actual_next = recent[i+pattern_length]
                            
                            # Determine what was expected based on historical data
                            if pattern in self.pattern_store:
                                continuations = self.pattern_store[pattern]
                                total_continuations = sum(continuations.values())
                                
                                if total_continuations >= self.repeat_threshold:
                                    # Find the most common continuation
                                    expected_next = continuations.most_common(1)[0][0]
                                    expected_prob = continuations[expected_next] / total_continuations
                                    
                                    # If the actual continuation was different from expected
                                    if actual_next != expected_next:
                                        # Calculate confidence in this interruption
                                        confidence = expected_prob
                                        return pattern, expected_next, actual_next, confidence
                            
                            break  # Only check the most recent occurrence
        
        return None
    
    def _get_followup_bet(self):
        """
        Get a follow-up bet after a pattern interruption.
        
        Returns:
            str: 'P', 'B', or 'SKIP' for the follow-up bet
        """
        if not self.recent_interruptions:
            return "SKIP"
        
        # Use the most recent interruption for follow-up
        pattern, interrupting_side = self.recent_interruptions[-1]
        
        # Analyze if the interrupting side tends to repeat after this pattern
        if pattern in self.pattern_store:
            continuations = self.pattern_store[pattern]
            total = sum(continuations.values())
            
            if total >= self.repeat_threshold:
                # If there's a clear tendency after this pattern was interrupted
                p_prob = continuations.get('P', 0) / total if total > 0 else 0.5
                b_prob = continuations.get('B', 0) / total if total > 0 else 0.5
                
                # Apply banker bias
                b_prob += self.banker_bias
                
                # Normalize
                total_prob = p_prob + b_prob
                if total_prob > 0:
                    p_prob /= total_prob
                    b_prob /= total_prob
                
                if p_prob > b_prob and p_prob > self.confidence_threshold:
                    return "P"
                elif b_prob > p_prob and b_prob > self.confidence_threshold:
                    return "B"
        
        # Default to betting on the same side that interrupted the pattern
        return interrupting_side
    
    def get_pattern_statistics(self):
        """
        Get statistics about detected patterns for debugging.
        
        Returns:
            dict: Statistics about patterns
        """
        stats = {
            "total_patterns": len(self.pattern_store),
            "top_patterns": [],
            "interruption_count": len(self.recent_interruptions)
        }
        
        # Find patterns with the strongest continuations
        strong_patterns = []
        for pattern, continuations in self.pattern_store.items():
            total = sum(continuations.values())
            if total >= self.repeat_threshold:
                # Find the dominant continuation
                most_common = continuations.most_common(1)
                if most_common:
                    dominant_outcome, count = most_common[0]
                    probability = count / total
                    
                    if probability >= 0.6:  # Only include patterns with clear tendencies
                        strong_patterns.append((pattern, dominant_outcome, probability))
        
        # Sort by probability
        strong_patterns.sort(key=lambda x: x[2], reverse=True)
        
        # Include top 5 patterns in stats
        stats["top_patterns"] = strong_patterns[:5]
        
        return stats