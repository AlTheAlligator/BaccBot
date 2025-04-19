"""
Hybrid strategy combining Frequency Analysis and Volatility Adaptive approaches.
Dynamically selects between strategies based on confidence levels and historical performance.
"""

import logging
from collections import deque
from typing import Dict, List, Tuple
from .frequency_analysis import FrequencyAnalysisStrategy
from .volatility_adaptive import VolatilityAdaptiveStrategy

logger = logging.getLogger(__name__)

class HybridFrequencyVolatilityStrategy:
    def __init__(self, params=None):
        
        params = params or {}
        # Initialize sub-strategies
        self.frequency_strategy = FrequencyAnalysisStrategy()
        self.volatility_strategy = VolatilityAdaptiveStrategy()
        
        # Performance tracking
        self.performance_window = params.get('performance_window', 88)
        self.frequency_performance = deque(maxlen=self.performance_window)
        self.volatility_performance = deque(maxlen=self.performance_window)
        
        # Strategy selection parameters
        self.min_confidence_diff = params.get('min_confidence_diff', 0.37326530612244896)
        self.performance_weight = params.get('performance_weight', 0.028163265306122447)
        self.confidence_weight = params.get('confidence_weight', 0.37326530612244896)
        
        # Tracking last decision for performance updates
        self.last_strategy = None
        self.last_prediction = None

    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet by comparing both strategies.
        
        Args:
            outcomes: List of previous outcomes ('P', 'B', 'T')
            
        Returns:
            str: Prediction ('P', 'B', or 'SKIP')
        """
        # Get predictions and confidence from both strategies
        freq_pred, freq_conf = self._get_frequency_prediction(outcomes)
        vol_pred, vol_conf = self._get_volatility_prediction(outcomes)
        
        # Calculate performance scores (win rate over window)
        freq_score = self._calculate_performance_score(self.frequency_performance)
        vol_score = self._calculate_performance_score(self.volatility_performance)
        
        # Calculate combined scores using both confidence and historical performance
        freq_combined = (freq_conf * self.confidence_weight + 
                        freq_score * self.performance_weight)
        vol_combined = (vol_conf * self.confidence_weight + 
                       vol_score * self.performance_weight)
        
        logger.debug(f"Frequency: pred={freq_pred}, conf={freq_conf:.3f}, score={freq_score:.3f}")
        logger.debug(f"Volatility: pred={vol_pred}, conf={vol_conf:.3f}, score={vol_score:.3f}")
        
        # Select strategy based on combined scores
        if abs(freq_combined - vol_combined) < self.min_confidence_diff:
            # If scores are close, use the one with better historical performance
            selected_strategy = "frequency" if freq_score > vol_score else "volatility"
            prediction = freq_pred if selected_strategy == "frequency" else vol_pred
        else:
            # Use strategy with higher combined score
            selected_strategy = "frequency" if freq_combined > vol_combined else "volatility"
            prediction = freq_pred if selected_strategy == "frequency" else vol_pred
        
        # Store decision for performance tracking
        self.last_strategy = selected_strategy
        self.last_prediction = prediction
        
        logger.debug(f"Selected {selected_strategy} strategy with prediction {prediction}")
        return prediction

    def _get_frequency_prediction(self, outcomes: List[str]) -> Tuple[str, float]:
        """Get prediction and confidence from frequency analysis strategy."""
        prediction = self.frequency_strategy.get_bet(outcomes)
        confidence = self.frequency_strategy.last_confidence
        return prediction, confidence

    def _get_volatility_prediction(self, outcomes: List[str]) -> Tuple[str, float]:
        """Get prediction and confidence from volatility adaptive strategy."""
        prediction = self.volatility_strategy.get_bet(outcomes)
        confidence = self.volatility_strategy.last_confidence
        return prediction, confidence

    def _calculate_performance_score(self, performance_history: deque) -> float:
        """Calculate win rate from performance history."""
        if not performance_history:
            return 0.5
        return sum(performance_history) / len(performance_history)

    def update_performance(self, outcome: str) -> None:
        """
        Update performance tracking after each hand.
        
        Args:
            outcome: Actual outcome of the hand ('P', 'B', 'T')
        """
        if self.last_prediction and outcome != 'T':
            is_win = self.last_prediction == outcome
            if self.last_strategy == "frequency":
                self.frequency_performance.append(float(is_win))
            else:
                self.volatility_performance.append(float(is_win))