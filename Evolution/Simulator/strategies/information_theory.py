"""
Information Theory strategy implementation.
"""

import logging
import numpy as np
from collections import Counter, defaultdict
import math
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class InformationTheoryStrategy(BaseStrategy):
    """
    Information Theory strategy that analyzes entropy and mutual information in outcomes.
    
    This strategy uses concepts from information theory to measure the randomness (entropy)
    of the outcome sequence and the mutual information between different time points.
    It identifies periods of low entropy (more predictable) and high mutual information
    (stronger dependencies) to make betting decisions.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Information Theory strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.entropy_window = params.get('entropy_window', 20)
        self.mutual_info_lag = params.get('mutual_info_lag', 3)
        self.entropy_threshold = params.get('entropy_threshold', 0.8)
        self.mutual_info_threshold = params.get('mutual_info_threshold', 0.1)
        self.min_samples = params.get('min_samples', 30)
        self.banker_bias = params.get('banker_bias', 0.01)
        
        # Initialize tracking variables
        self.numeric_history = []  # 1 for Player, 0 for Banker
        self.pattern_frequencies = defaultdict(lambda: {'P': 0, 'B': 0})
        
        logger.info(f"Initialized Information Theory strategy with entropy_window={self.entropy_window}")
    
    def _outcomes_to_numeric(self, outcomes):
        """
        Convert outcome strings to numeric values for analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            list: Numeric values (1 for P, 0 for B)
        """
        return [1 if o == 'P' else 0 for o in outcomes]
    
    def _calculate_entropy(self, sequence):
        """
        Calculate Shannon entropy of a sequence.
        
        Args:
            sequence: Numeric sequence
            
        Returns:
            float: Shannon entropy (normalized to [0,1])
        """
        if not sequence:
            return 1.0  # Maximum entropy (complete randomness) for empty sequence
        
        # Count occurrences
        counts = Counter(sequence)
        total = len(sequence)
        
        # Calculate entropy
        entropy = 0
        for count in counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)
        
        # Normalize to [0,1] - binary entropy max is 1.0
        return entropy
    
    def _calculate_conditional_entropy(self, sequence, lag):
        """
        Calculate conditional entropy H(X_t | X_{t-lag}).
        
        Args:
            sequence: Numeric sequence
            lag: Time lag
            
        Returns:
            float: Conditional entropy
        """
        if len(sequence) <= lag:
            return 1.0  # Maximum entropy for insufficient data
        
        # Create pairs of (X_{t-lag}, X_t)
        pairs = [(sequence[i-lag], sequence[i]) for i in range(lag, len(sequence))]
        
        # Count joint occurrences
        joint_counts = Counter(pairs)
        total_pairs = len(pairs)
        
        # Count marginal occurrences
        marginal_counts = Counter([pair[0] for pair in pairs])
        
        # Calculate conditional entropy
        cond_entropy = 0
        for (x_prev, x_curr), joint_count in joint_counts.items():
            joint_prob = joint_count / total_pairs
            marginal_prob = marginal_counts[x_prev] / total_pairs
            cond_entropy -= joint_prob * math.log2(joint_prob / marginal_prob)
        
        # Normalize to [0,1]
        return min(1.0, cond_entropy)
    
    def _calculate_mutual_information(self, sequence, lag):
        """
        Calculate mutual information I(X_t; X_{t-lag}).
        
        Args:
            sequence: Numeric sequence
            lag: Time lag
            
        Returns:
            float: Mutual information
        """
        if len(sequence) <= lag:
            return 0.0  # No mutual information for insufficient data
        
        # Calculate entropy
        entropy = self._calculate_entropy(sequence[lag:])
        
        # Calculate conditional entropy
        cond_entropy = self._calculate_conditional_entropy(sequence, lag)
        
        # Mutual information = H(X_t) - H(X_t | X_{t-lag})
        mutual_info = entropy - cond_entropy
        
        return max(0.0, mutual_info)  # Ensure non-negative
    
    def _update_pattern_frequencies(self, outcomes):
        """
        Update pattern frequency counts.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Process patterns of different lengths
        for length in range(1, self.mutual_info_lag + 1):
            if len(outcomes) <= length:
                continue
                
            # Process each pattern
            for i in range(len(outcomes) - length):
                pattern = tuple(outcomes[i:i+length])
                next_outcome = outcomes[i+length]
                
                # Update pattern store
                self.pattern_frequencies[pattern][next_outcome] += 1
    
    def _predict_from_pattern(self, pattern, outcomes):
        """
        Predict next outcome based on pattern frequencies.
        
        Args:
            pattern: Current pattern
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted next outcome ('P' or 'B')
        """
        if pattern in self.pattern_frequencies:
            counts = self.pattern_frequencies[pattern]
            p_count = counts['P']
            b_count = counts['B']
            
            if p_count + b_count > 0:
                p_prob = p_count / (p_count + b_count)
                b_prob = b_count / (p_count + b_count)
                
                # Apply banker bias
                b_prob += self.banker_bias
                
                # Normalize
                total = p_prob + b_prob
                p_prob /= total
                b_prob /= total
                
                return 'P' if p_prob > b_prob else 'B'
        
        # Default to most frequent outcome in recent history
        recent = outcomes[-self.entropy_window:]
        p_count = recent.count('P')
        b_count = recent.count('B')
        
        # Apply banker bias
        b_count += b_count * self.banker_bias
        
        return 'P' if p_count > b_count else 'B'
    
    def _calculate_transfer_entropy(self, sequence, lag):
        """
        Calculate transfer entropy (directional information flow).
        
        Args:
            sequence: Numeric sequence
            lag: Time lag
            
        Returns:
            float: Transfer entropy
        """
        if len(sequence) <= 2*lag:
            return 0.0  # No transfer entropy for insufficient data
        
        # Calculate conditional entropy H(X_t | X_{t-1})
        cond_entropy1 = self._calculate_conditional_entropy(sequence, 1)
        
        # Calculate conditional entropy H(X_t | X_{t-1}, X_{t-lag})
        # This is an approximation - full calculation would require joint distributions
        cond_entropy2 = self._calculate_conditional_entropy(sequence, lag)
        
        # Transfer entropy = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, X_{t-lag})
        # This is an approximation
        transfer_entropy = cond_entropy1 - min(cond_entropy1, cond_entropy2)
        
        return max(0.0, transfer_entropy)  # Ensure non-negative
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using information theory analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)
        
        # Always start from game 7
        if len(outcomes) < 7:
            return 'B'  # Default to Banker for initial games
        
        # Filter out ties
        filtered = [o for o in outcomes if o in ['P', 'B']]
        
        # Not enough data - use simple frequency analysis
        if len(filtered) < self.min_samples:
            p_count = filtered.count('P')
            b_count = filtered.count('B')
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'
        
        # Convert to numeric for entropy calculations
        numeric = self._outcomes_to_numeric(filtered)
        self.numeric_history = numeric
        
        # Update pattern frequencies
        self._update_pattern_frequencies(filtered)
        
        # Calculate entropy of recent outcomes
        recent = numeric[-self.entropy_window:]
        entropy = self._calculate_entropy(recent)
        
        # Calculate mutual information between current and lagged outcomes
        mutual_info = self._calculate_mutual_information(numeric, self.mutual_info_lag)
        
        # Calculate transfer entropy
        transfer_entropy = self._calculate_transfer_entropy(numeric, self.mutual_info_lag)
        
        logger.debug(f"Entropy: {entropy:.3f}, Mutual Info: {mutual_info:.3f}, Transfer Entropy: {transfer_entropy:.3f}")
        
        # Make decision based on information theory metrics
        if entropy < self.entropy_threshold:
            # Low entropy (more predictable) - use frequency analysis
            p_count = recent.count(1)
            b_count = recent.count(0)
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            # Normalize
            total = p_count + b_count
            p_prob = p_count / total
            b_prob = b_count / total
            
            logger.debug(f"Low entropy mode: P={p_prob:.3f}, B={b_prob:.3f}")
            
            # Return the more frequent outcome
            return 'P' if p_prob > b_prob else 'B'
        elif mutual_info > self.mutual_info_threshold:
            # High mutual information - use pattern matching
            pattern = tuple(filtered[-self.mutual_info_lag:])
            next_outcome = self._predict_from_pattern(pattern, filtered)
            
            logger.debug(f"High mutual info mode: pattern={pattern}, prediction={next_outcome}")
            
            return next_outcome
        else:
            # High randomness - default to Banker with slight edge
            logger.debug("High randomness mode: defaulting to Banker")
            return 'B'
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        if len(self.numeric_history) >= self.entropy_window:
            recent = self.numeric_history[-self.entropy_window:]
            entropy = self._calculate_entropy(recent)
        else:
            entropy = 1.0
            
        if len(self.numeric_history) >= self.mutual_info_lag:
            mutual_info = self._calculate_mutual_information(self.numeric_history, self.mutual_info_lag)
        else:
            mutual_info = 0.0
            
        return {
            "entropy": entropy,
            "mutual_info": mutual_info,
            "entropy_threshold": self.entropy_threshold,
            "mutual_info_threshold": self.mutual_info_threshold,
            "pattern_count": len(self.pattern_frequencies)
        }
