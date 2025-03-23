"""
Adaptive Bias strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class AdaptiveBiasStrategy(BaseStrategy):
    """
    Adaptively learn table bias using weighted recent outcomes.
    
    This strategy uses both short-term and long-term trend analysis to 
    detect biases in the game outcomes and adjust betting accordingly.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate window size
        self.window_size = self.params.get('window_size', 20)
        if not isinstance(self.window_size, int) or self.window_size < 5:
            raise ValueError("window_size must be an integer >= 5")
            
        # Validate weight_recent
        self.weight_recent = self.params.get('weight_recent', 2.0)
        if not isinstance(self.weight_recent, (int, float)) or self.weight_recent < 1:
            raise ValueError("weight_recent must be a number >= 1")
            
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tWindow size: {self.window_size}"
                   f"\n\tWeight recent: {self.weight_recent}")
    
    def get_bet(self, outcomes):
        """
        Determine next bet based on adaptive bias analysis
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.window_size:
            return 'B'  # Default when not enough history
        
        # Calculate long-term bias from historical data
        total_p = historical_outcomes.count('P')
        total_b = historical_outcomes.count('B')
        total = total_p + total_b
        long_term_bias = total_p / total if total > 0 else 0.5
        
        # Calculate short-term bias with weighted recent outcomes
        recent_outcomes = historical_outcomes[-self.window_size:]
        weighted_p = 0
        weighted_b = 0
        
        for i, outcome in enumerate(recent_outcomes):
            # Calculate weight - more recent outcomes have higher weights
            weight = 1.0 + (self.weight_recent - 1.0) * (i / (self.window_size - 1))
            if outcome == 'P':
                weighted_p += weight
            elif outcome == 'B':
                weighted_b += weight
        
        total_weighted = weighted_p + weighted_b
        short_term_bias = weighted_p / total_weighted if total_weighted > 0 else 0.5
        
        # Combine biases (60% weight to short-term, 40% to long-term)
        combined_bias = 0.6 * short_term_bias + 0.4 * long_term_bias
        
        # Calculate trend from most recent outcomes
        last_5 = historical_outcomes[-5:] if len(historical_outcomes) >= 5 else historical_outcomes
        recent_p = last_5.count('P')
        recent_trend = recent_p / len(last_5) if last_5 else 0.5
        
        # If we detect a strong trend in either direction, adjust our bias
        if recent_trend > 0.7 or recent_trend < 0.3:
            combined_bias = (combined_bias + recent_trend) / 2
            
        # Make betting decision based on combined bias
        return 'P' if combined_bias > 0.5 else 'B'