"""
Hybrid Pattern strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HybridPatternStrategy(BaseStrategy):
    """
    Hybrid strategy that uses pattern detection to choose between
    PPP and BBB modes, then applies the original strategy logic.
    
    Note: This strategy is special as it requires integration with the simulator's
    mode switching logic. The get_bet() method returns None as the actual betting
    decision is handled by the GameSimulator class.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate parameters
        self.pattern_length = self.params.get('pattern_length', 6)
        if not isinstance(self.pattern_length, int) or self.pattern_length < 3:
            raise ValueError("pattern_length must be an integer >= 3")
            
        self.alternating_boost = self.params.get('alternating_boost', 1.0)
        if not isinstance(self.alternating_boost, (int, float)) or self.alternating_boost <= 0:
            raise ValueError("alternating_boost must be a positive number")
            
        self.trend_threshold = self.params.get('trend_threshold', 0.7)
        if not isinstance(self.trend_threshold, (int, float)) or not 0 < self.trend_threshold < 1:
            raise ValueError("trend_threshold must be between 0 and 1")
            
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tPattern length: {self.pattern_length}"
                   f"\n\tAlternating boost: {self.alternating_boost}"
                   f"\n\tTrend threshold: {self.trend_threshold}")
        
    def get_bet(self, outcomes):
        """
        This strategy requires special handling in the simulator as it integrates with
        the original strategy's mode switching logic.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            str: The betting decision based on pattern analysis
        """
        if not outcomes:
            return 'B'  # Default to banker with no history
            
        bias_strength, preferred_mode = self.analyze_pattern(outcomes)
        
        # Use pattern analysis to influence bet
        if preferred_mode == "BBB":
            return "P"  # In BBB mode, bet on Player
        elif preferred_mode == "PPP":
            return "B"  # In PPP mode, bet on Banker
        else:
            # If no strong pattern, use current mode
            return "B" if self.simulator.current_mode == "PPP" else "P"
        
    def analyze_pattern(self, outcomes):
        """
        Analyze current pattern to help determine optimal mode.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            tuple: (bias_strength, preferred_mode)
                  bias_strength: float between 0 and 1
                  preferred_mode: 'PPP' or 'BBB'
        """
        if len(outcomes) < self.pattern_length:
            return 0.5, self.simulator.current_mode
            
        # Get recent non-tie outcomes
        historical = self._get_historical_outcomes(outcomes)
        if len(historical) < self.pattern_length:
            return 0.5, self.simulator.current_mode
            
        recent = historical[-self.pattern_length:]
        
        # Calculate bias
        p_count = recent.count('P')
        b_count = recent.count('B')
        total = p_count + b_count
        
        if total == 0:
            return 0.5, self.simulator.current_mode
            
        p_ratio = p_count / total
        
        # Check for alternating pattern and adjust ratio
        alternating, last_value = self._is_alternating(historical, min(4, self.pattern_length))
        if alternating:
            if last_value == 'P':
                p_ratio = min(1.0, p_ratio * self.alternating_boost)
            else:
                p_ratio = max(0.0, p_ratio / self.alternating_boost)
                
        # Calculate bias strength and determine mode
        bias_strength = abs(p_ratio - 0.5) * 2  # Convert to 0-1 scale
        
        if p_ratio > self.trend_threshold:
            return bias_strength, "BBB"  # Strong player bias suggests BBB mode
        elif p_ratio < (1 - self.trend_threshold):
            return bias_strength, "PPP"  # Strong banker bias suggests PPP mode
            
        return bias_strength, self.simulator.current_mode