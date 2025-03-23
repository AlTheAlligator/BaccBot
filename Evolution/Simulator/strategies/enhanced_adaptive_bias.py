"""
Enhanced Adaptive Bias strategy implementation.
"""

import logging
import numpy as np
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class EnhancedAdaptiveBiasStrategy(BaseStrategy):
    """
    Enhanced adaptive bias strategy with multiple improvements.
    
    This strategy uses multiple timeframes for analysis, pattern recognition,
    and adaptive confidence thresholds to make more informed decisions.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate window sizes
        self.window_sizes = self.params.get('window_sizes', [5, 10, 20])
        if not isinstance(self.window_sizes, list) or len(self.window_sizes) == 0:
            raise ValueError("window_sizes must be a non-empty list")
        if not all(isinstance(w, int) and w > 0 for w in self.window_sizes):
            raise ValueError("all window sizes must be positive integers")
        if not all(self.window_sizes[i] < self.window_sizes[i+1] for i in range(len(self.window_sizes)-1)):
            raise ValueError("window sizes must be in ascending order")
            
        # Validate other parameters
        self.base_weight_recent = self.params.get('base_weight_recent', 2.5)
        if not isinstance(self.base_weight_recent, (int, float)) or self.base_weight_recent < 1:
            raise ValueError("base_weight_recent must be a number >= 1")
            
        self.confidence_threshold = self.params.get('confidence_threshold', 0.65)
        if not isinstance(self.confidence_threshold, (int, float)) or not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
            
        self.adaptation_rate = self.params.get('adaptation_rate', 0.15)
        if not isinstance(self.adaptation_rate, (int, float)) or not 0 < self.adaptation_rate < 1:
            raise ValueError("adaptation_rate must be between 0 and 1")
            
        # Initialize performance tracking state
        self._enhanced_adaptive_state = {
            'performance_window': [],
            'confidence_history': [],
            'last_prediction': None,
            'current_weight': self.base_weight_recent,
            'loss_streak': 0,
            'win_streak': 0
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tWindow sizes: {self.window_sizes}"
                   f"\n\tBase weight recent: {self.base_weight_recent}"
                   f"\n\tConfidence threshold: {self.confidence_threshold}"
                   f"\n\tAdaptation rate: {self.adaptation_rate}")

    def get_bet(self, outcomes):
        """
        Determine next bet using enhanced adaptive bias analysis
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < max(self.window_sizes):
            return 'B'  # Default when not enough history
            
        state = self._enhanced_adaptive_state
        
        # Update performance metrics if we have a last prediction
        if state['last_prediction'] and len(historical_outcomes) > 0:
            last_outcome = historical_outcomes[-1]
            is_win = (state['last_prediction'] == last_outcome)
            
            state['performance_window'].append(1 if is_win else 0)
            if len(state['performance_window']) > 20:
                state['performance_window'].pop(0)
                
            if is_win:
                state['win_streak'] += 1
                state['loss_streak'] = 0
            else:
                state['loss_streak'] += 1
                state['win_streak'] = 0
        
        # Calculate win rate from recent performance
        win_rate = sum(state['performance_window']) / len(state['performance_window']) if state['performance_window'] else 0.5
        
        # Dynamically adjust weights based on performance
        performance_factor = (win_rate - 0.5) * 2  # Scale to [-1, 1]
        state['current_weight'] = self.base_weight_recent * (1 + performance_factor * self.adaptation_rate)
        
        # Multi-timeframe analysis
        window_biases = []
        for window_size in self.window_sizes:
            if len(historical_outcomes) >= window_size:
                recent_outcomes = historical_outcomes[-window_size:]
                weighted_p = 0
                weighted_b = 0
                
                for i, outcome in enumerate(recent_outcomes):
                    weight = 1.0 + (state['current_weight'] - 1.0) * (i / (window_size - 1))
                    if outcome == 'P':
                        weighted_p += weight
                    elif outcome == 'B':
                        weighted_b += weight
                
                total_weighted = weighted_p + weighted_b
                if total_weighted > 0:
                    bias = weighted_p / total_weighted
                    window_biases.append(bias)
        
        if not window_biases:
            return 'B'  # Default when no valid biases
        
        # Combine biases with emphasis on shorter timeframes
        weights = [0.5, 0.3, 0.2]  # Weights for different timeframes (short to long)
        combined_bias = sum(b * w for b, w in zip(window_biases, weights[:len(window_biases)])) / sum(weights[:len(window_biases)])
        
        # Pattern recognition
        pattern_adjustment = 0
        if len(historical_outcomes) >= 5:
            last_5 = historical_outcomes[-5:]
            
            # Check for alternating pattern
            is_alternating = True
            for i in range(1, len(last_5)):
                if last_5[i] == last_5[i-1]:
                    is_alternating = False
                    break
                    
            if is_alternating:
                last_outcome = last_5[-1]
                if last_outcome == 'P':
                    pattern_adjustment = -0.1  # Adjust toward B
                else:
                    pattern_adjustment = 0.1   # Adjust toward P
            
            # Check for streaks
            streak_length = 1
            for i in range(len(last_5)-2, -1, -1):
                if last_5[i] == last_5[-1]:
                    streak_length += 1
                else:
                    break
                    
            if streak_length >= 3:
                if last_5[-1] == 'P':
                    pattern_adjustment -= 0.15  # Long P streak, adjust toward B
                else:
                    pattern_adjustment += 0.15  # Long B streak, adjust toward P
        
        # Apply pattern adjustment
        combined_bias += pattern_adjustment
        
        # Loss recovery adaptation
        if state['loss_streak'] >= 2:
            # Be more conservative after losses
            combined_bias = (combined_bias * 0.7) + (0.5 * 0.3)
        
        # Calculate decision confidence
        confidence = abs(combined_bias - 0.5) * 2  # Scale to [0, 1]
        state['confidence_history'].append(confidence)
        if len(state['confidence_history']) > 10:
            state['confidence_history'].pop(0)
        
        # Store prediction for performance tracking
        decision = 'P' if combined_bias > 0.5 else 'B'
        state['last_prediction'] = decision
        
        # High confidence threshold override
        avg_confidence = sum(state['confidence_history']) / len(state['confidence_history']) if state['confidence_history'] else 0
        if confidence > self.confidence_threshold and avg_confidence > self.confidence_threshold:
            return 'P' if combined_bias > 0.5 else 'B'
        
        # Normal decision with mean reversion consideration
        if combined_bias > 0.55:
            return 'P'
        elif combined_bias < 0.45:
            return 'B'
        else:
            # In uncertain territory, favor banker due to lower commission
            return 'B'