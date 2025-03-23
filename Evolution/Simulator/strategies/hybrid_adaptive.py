"""
Hybrid Adaptive strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HybridAdaptiveStrategy(BaseStrategy):
    """
    Hybrid strategy that uses the original mode-switching approach but with
    adaptive bias detection to determine the optimal mode.
    
    This strategy combines the original BBB/PPP approach with adaptive bias
    analysis to dynamically adjust the mode based on detected table bias.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate parameters
        self.window_size = self.params.get('window_size', 30)
        if not isinstance(self.window_size, int) or self.window_size < 5:
            raise ValueError("window_size must be an integer >= 5")
            
        self.weight_recent = self.params.get('weight_recent', 2.5)
        if not isinstance(self.weight_recent, (int, float)) or self.weight_recent < 1:
            raise ValueError("weight_recent must be a number >= 1")
            
        # Initialize state
        self._hybrid_state = {
            'performance_window': [],
            'recent_biases': [],
            'suggested_mode': None,
            'mode_confidence': 0.0,
            'last_bias': 0.5
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tWindow size: {self.window_size}"
                   f"\n\tWeight recent: {self.weight_recent}")
        
    def get_bet(self, outcomes):
        """
        Determine next bet using hybrid approach with adaptive bias.
        
        This implementation analyzes table bias to suggest optimal mode,
        but the actual betting decision is handled by the simulator's
        mode switching logic.
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.window_size:
            return None  # Let simulator use default mode
            
        state = self._hybrid_state
        
        # Calculate weighted bias from recent outcomes
        recent = historical_outcomes[-self.window_size:]
        weighted_p = 0
        weighted_b = 0
        total_weight = 0
        
        for i, outcome in enumerate(recent):
            # More recent outcomes have higher weights
            weight = 1.0 + (self.weight_recent - 1.0) * (i / (self.window_size - 1))
            total_weight += weight
            
            if outcome == 'P':
                weighted_p += weight
            elif outcome == 'B':
                weighted_b += weight
                
        # Calculate bias (0 = strong B bias, 1 = strong P bias)
        current_bias = weighted_p / total_weight if total_weight > 0 else 0.5
        state['recent_biases'].append(current_bias)
        if len(state['recent_biases']) > 5:
            state['recent_biases'].pop(0)
            
        # Analyze bias trend
        avg_bias = sum(state['recent_biases']) / len(state['recent_biases'])
        bias_trend = avg_bias - state['last_bias']
        state['last_bias'] = avg_bias
        
        # Determine suggested mode based on bias
        if avg_bias > 0.6 or (avg_bias > 0.55 and bias_trend > 0.02):
            # Strong or increasing P bias - suggest BBB mode to bet against it
            state['suggested_mode'] = "BBB"
            state['mode_confidence'] = min(1.0, (avg_bias - 0.5) * 4)
        elif avg_bias < 0.4 or (avg_bias < 0.45 and bias_trend < -0.02):
            # Strong or increasing B bias - suggest PPP mode to bet against it
            state['suggested_mode'] = "PPP"
            state['mode_confidence'] = min(1.0, (0.5 - avg_bias) * 4)
        else:
            # No strong bias - stick with current mode
            state['suggested_mode'] = self.simulator.current_mode
            state['mode_confidence'] = 0.5
            
        # The simulator will use suggested_mode and mode_confidence
        # to help determine mode switches
        return None