"""
Transfer Learning strategy implementation.
"""

import logging
import numpy as np
from collections import defaultdict
import random
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TransferLearningStrategy(BaseStrategy):
    """
    Transfer Learning strategy for baccarat betting.
    
    This strategy applies knowledge learned from previous shoes to the current shoe,
    adapting the model as new data becomes available while preserving useful patterns.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Transfer Learning strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.min_samples = params.get('min_samples', 20)  # Minimum samples before making predictions
        self.pattern_length = params.get('pattern_length', 3)  # Length of patterns to consider
        self.confidence_threshold = params.get('confidence_threshold', 0.55)  # Threshold for making a bet
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.transfer_weight = params.get('transfer_weight', 0.3)  # Weight given to previous knowledge
        self.adaptation_rate = params.get('adaptation_rate', 0.1)  # Rate at which to adapt to new data
        self.max_shoes = params.get('max_shoes', 5)  # Maximum number of shoes to remember
        
        # Knowledge base
        self.global_knowledge = {
            'patterns': defaultdict(lambda: {'P': 0, 'B': 0}),
            'transitions': defaultdict(lambda: {'P': 0, 'B': 0}),
            'global_frequencies': {'P': 0, 'B': 0}
        }
        
        # Current shoe knowledge
        self.current_knowledge = {
            'patterns': defaultdict(lambda: {'P': 0, 'B': 0}),
            'transitions': defaultdict(lambda: {'P': 0, 'B': 0}),
            'global_frequencies': {'P': 0, 'B': 0}
        }
        
        # Shoe history
        self.shoe_history = []
        self.current_shoe_outcomes = []
        self.shoe_boundary_detected = False
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logger.info(f"Initialized Transfer Learning strategy with transfer_weight={self.transfer_weight}")
    
    def _detect_shoe_boundary(self, outcomes):
        """
        Detect if a new shoe has started.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            bool: Whether a new shoe has been detected
        """
        # Simple heuristic: if there's a gap of more than 20 hands, assume it's a new shoe
        if len(self.current_shoe_outcomes) > 0 and len(outcomes) - len(self.current_shoe_outcomes) > 20:
            return True
        
        # Another heuristic: if the total outcomes has decreased, it's a new shoe
        if len(outcomes) < len(self.current_shoe_outcomes):
            return True
        
        return False
    
    def _update_knowledge(self, outcomes):
        """
        Update knowledge base with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update global frequencies
        for outcome in outcomes:
            self.current_knowledge['global_frequencies'][outcome] += 1
        
        # Update first-order transitions
        if len(outcomes) >= 2:
            for i in range(len(outcomes) - 1):
                current = outcomes[i]
                next_outcome = outcomes[i + 1]
                self.current_knowledge['transitions'][current][next_outcome] += 1
        
        # Update pattern-based transitions
        if len(outcomes) >= self.pattern_length + 1:
            for i in range(len(outcomes) - self.pattern_length):
                pattern = tuple(outcomes[i:i+self.pattern_length])
                next_outcome = outcomes[i+self.pattern_length]
                self.current_knowledge['patterns'][pattern][next_outcome] += 1
    
    def _save_current_shoe(self):
        """Save the current shoe to history and reset current knowledge."""
        if len(self.current_shoe_outcomes) > 0:
            # Add current shoe to history
            self.shoe_history.append({
                'outcomes': self.current_shoe_outcomes.copy(),
                'knowledge': {
                    'patterns': dict(self.current_knowledge['patterns']),
                    'transitions': dict(self.current_knowledge['transitions']),
                    'global_frequencies': dict(self.current_knowledge['global_frequencies'])
                }
            })
            
            # Limit the number of shoes in history
            if len(self.shoe_history) > self.max_shoes:
                self.shoe_history.pop(0)
            
            # Reset current shoe
            self.current_shoe_outcomes = []
            self.current_knowledge = {
                'patterns': defaultdict(lambda: {'P': 0, 'B': 0}),
                'transitions': defaultdict(lambda: {'P': 0, 'B': 0}),
                'global_frequencies': {'P': 0, 'B': 0}
            }
            
            logger.info(f"Saved shoe to history. Total shoes: {len(self.shoe_history)}")
    
    def _transfer_knowledge(self):
        """Transfer knowledge from previous shoes to global knowledge."""
        # Reset global knowledge
        self.global_knowledge = {
            'patterns': defaultdict(lambda: {'P': 0, 'B': 0}),
            'transitions': defaultdict(lambda: {'P': 0, 'B': 0}),
            'global_frequencies': {'P': 0, 'B': 0}
        }
        
        # Combine knowledge from all shoes in history
        for shoe in self.shoe_history:
            # Global frequencies
            for outcome, count in shoe['knowledge']['global_frequencies'].items():
                self.global_knowledge['global_frequencies'][outcome] += count
            
            # Transitions
            for state, transitions in shoe['knowledge']['transitions'].items():
                for outcome, count in transitions.items():
                    self.global_knowledge['transitions'][state][outcome] += count
            
            # Patterns
            for pattern, outcomes in shoe['knowledge']['patterns'].items():
                for outcome, count in outcomes.items():
                    self.global_knowledge['patterns'][pattern][outcome] += count
        
        logger.info("Transferred knowledge from previous shoes to global knowledge")
    
    def _get_prediction(self, outcomes):
        """
        Get prediction based on current and global knowledge.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (prediction, confidence)
        """
        # Not enough data
        if len(outcomes) < self.pattern_length:
            return 'B', 0.51  # Default to banker with minimal confidence
        
        # Get current pattern
        current_pattern = tuple(outcomes[-self.pattern_length:])
        
        # Check if pattern exists in current knowledge
        current_pattern_counts = self.current_knowledge['patterns'].get(current_pattern, {'P': 0, 'B': 0})
        current_total = current_pattern_counts['P'] + current_pattern_counts['B']
        
        # Check if pattern exists in global knowledge
        global_pattern_counts = self.global_knowledge['patterns'].get(current_pattern, {'P': 0, 'B': 0})
        global_total = global_pattern_counts['P'] + global_pattern_counts['B']
        
        # Calculate probabilities
        if current_total > 0 and global_total > 0:
            # Combine current and global knowledge
            current_p_prob = current_pattern_counts['P'] / current_total
            current_b_prob = current_pattern_counts['B'] / current_total
            
            global_p_prob = global_pattern_counts['P'] / global_total
            global_b_prob = global_pattern_counts['B'] / global_total
            
            # Weight based on transfer weight
            p_prob = (1 - self.transfer_weight) * current_p_prob + self.transfer_weight * global_p_prob
            b_prob = (1 - self.transfer_weight) * current_b_prob + self.transfer_weight * global_b_prob
        elif current_total > 0:
            # Only current knowledge available
            p_prob = current_pattern_counts['P'] / current_total
            b_prob = current_pattern_counts['B'] / current_total
        elif global_total > 0:
            # Only global knowledge available
            p_prob = global_pattern_counts['P'] / global_total
            b_prob = global_pattern_counts['B'] / global_total
        else:
            # Fall back to global frequencies
            current_freqs = self.current_knowledge['global_frequencies']
            global_freqs = self.global_knowledge['global_frequencies']
            
            current_total = current_freqs['P'] + current_freqs['B']
            global_total = global_freqs['P'] + global_freqs['B']
            
            if current_total > 0 and global_total > 0:
                current_p_prob = current_freqs['P'] / current_total
                current_b_prob = current_freqs['B'] / current_total
                
                global_p_prob = global_freqs['P'] / global_total
                global_b_prob = global_freqs['B'] / global_total
                
                p_prob = (1 - self.transfer_weight) * current_p_prob + self.transfer_weight * global_p_prob
                b_prob = (1 - self.transfer_weight) * current_b_prob + self.transfer_weight * global_b_prob
            elif current_total > 0:
                p_prob = current_freqs['P'] / current_total
                b_prob = current_freqs['B'] / current_total
            elif global_total > 0:
                p_prob = global_freqs['P'] / global_total
                b_prob = global_freqs['B'] / global_total
            else:
                # No data at all
                p_prob = 0.5
                b_prob = 0.5
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize
        total_prob = p_prob + b_prob
        p_prob /= total_prob
        b_prob /= total_prob
        
        # Determine prediction and confidence
        if p_prob > b_prob:
            return 'P', p_prob
        else:
            return 'B', b_prob
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using transfer learning.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)
        
        # Filter out ties
        filtered_outcomes = [o for o in outcomes if o in ['P', 'B']]
        
        # Check for shoe boundary
        if self._detect_shoe_boundary(filtered_outcomes):
            self._save_current_shoe()
            self._transfer_knowledge()
            self.shoe_boundary_detected = True
        
        # Update current shoe outcomes
        self.current_shoe_outcomes = filtered_outcomes.copy()
        
        # Update knowledge
        self._update_knowledge(filtered_outcomes)
        
        # Not enough data - default to Banker (slight edge due to commission)
        if len(filtered_outcomes) < self.min_samples:
            return 'B'
        
        # Get prediction
        bet, confidence = self._get_prediction(filtered_outcomes)
        
        # Always return a bet (no skipping)
        return bet
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "shoes_in_history": len(self.shoe_history),
            "current_shoe_length": len(self.current_shoe_outcomes),
            "global_patterns": len(self.global_knowledge['patterns']),
            "current_patterns": len(self.current_knowledge['patterns']),
            "transfer_weight": self.transfer_weight,
            "shoe_boundary_detected": self.shoe_boundary_detected
        }
