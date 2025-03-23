"""
Pattern-Based strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class PatternBasedStrategy(BaseStrategy):
    """
    Look for specific patterns and bet accordingly.
    This strategy analyzes patterns in recent outcomes to predict future results.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.pattern_length = self.params.get('pattern_length', 4)
    
    def get_bet(self, outcomes):
        """
        Determine next bet based on pattern analysis
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.pattern_length:
            return 'B'  # Default when not enough history
        
        # Get the last pattern_length outcomes
        current_pattern = historical_outcomes[-self.pattern_length:]
        
        # Count frequency in current pattern
        p_count = current_pattern.count('P')
        b_count = current_pattern.count('B')
        total = p_count + b_count
        
        # If heavily biased pattern, bet against the bias
        if total > 0:
            if p_count / total >= 0.75:
                return 'B'
            elif b_count / total >= 0.75:
                return 'P'
        
        # Look for alternating patterns
        alternating = True
        for i in range(1, len(current_pattern)):
            if current_pattern[i] == current_pattern[i-1]:
                alternating = False
                break
        
        if alternating:
            return 'P' if current_pattern[-1] == 'B' else 'B'
        
        # Default to countering last outcome
        return 'B' if current_pattern[-1] == 'P' else 'P'