"""
Adaptive Momentum strategy implementation.
"""

import logging
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class AdaptiveMomentumStrategy(BaseStrategy):
    """
    Adaptive Momentum strategy that combines technical analysis with adaptive learning.
    
    This strategy calculates multiple momentum indicators over different time windows
    and uses adaptive weights that adjust based on recent performance. It also includes
    a dynamic threshold system and a mean-reversion component.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Adaptive Momentum strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.window_sizes = params.get('window_sizes', [5, 10, 20, 30, 50])
        self.weight_adaptation_rate = params.get('weight_adaptation_rate', 0.1)
        self.threshold_adaptation_rate = params.get('threshold_adaptation_rate', 0.05)
        self.mean_reversion_threshold = params.get('mean_reversion_threshold', 0.7)
        self.initial_threshold = params.get('initial_threshold', 0.55)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.memory_length = params.get('memory_length', 100)
        
        # Initialize weights for different window sizes
        self.weights = np.ones(len(self.window_sizes)) / len(self.window_sizes)
        
        # Initialize dynamic threshold
        self.threshold = self.initial_threshold
        
        # Initialize memory
        self.outcome_memory = deque(maxlen=self.memory_length)
        self.prediction_memory = deque(maxlen=self.memory_length)
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        self.consecutive_losses = 0
        
        # Mean reversion mode
        self.mean_reversion_mode = False
        self.mean_reversion_counter = 0
    
    def _calculate_momentum(self, outcomes, window_size):
        """
        Calculate momentum indicator for a specific window size.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            window_size: Size of window to analyze
            
        Returns:
            float: Momentum indicator between -1 and 1
        """
        if len(outcomes) < window_size:
            return 0.0
        
        # Get window of outcomes
        window = outcomes[-window_size:]
        
        # Convert to numeric (1 for P, -1 for B)
        numeric = [1 if o == 'P' else -1 for o in window]
        
        # Calculate simple momentum (sum of values)
        simple_momentum = sum(numeric) / window_size
        
        # Calculate weighted momentum (more recent outcomes have higher weight)
        weights = np.linspace(1, 2, window_size)
        weights = weights / np.sum(weights)  # Normalize
        weighted_momentum = np.sum(np.array(numeric) * weights)
        
        # Calculate rate of change
        if len(numeric) >= 2:
            rate_of_change = numeric[-1] - numeric[0]
        else:
            rate_of_change = 0
        
        # Combine indicators
        momentum = 0.5 * simple_momentum + 0.3 * weighted_momentum + 0.2 * rate_of_change
        
        # Ensure result is between -1 and 1
        return max(-1, min(1, momentum))
    
    def _update_weights(self, prediction, actual):
        """
        Update weights based on prediction accuracy.
        
        Args:
            prediction: Predicted outcome ('P' or 'B')
            actual: Actual outcome ('P' or 'B')
        """
        # Convert to numeric
        pred_numeric = 1 if prediction == 'P' else -1
        actual_numeric = 1 if actual == 'P' else -1
        
        # Calculate error
        error = actual_numeric - pred_numeric
        
        # Update weights based on how well each momentum indicator predicted
        for i, window_size in enumerate(self.window_sizes):
            if len(self.outcome_memory) >= window_size:
                # Calculate what this indicator would have predicted
                momentum = self._calculate_momentum(list(self.outcome_memory)[:-1], window_size)
                indicator_pred = 1 if momentum > 0 else -1
                
                # Calculate indicator error
                indicator_error = actual_numeric - indicator_pred
                
                # Update weight (increase if correct, decrease if wrong)
                if indicator_error == 0:  # Correct prediction
                    self.weights[i] += self.weight_adaptation_rate * (1 - self.weights[i])
                else:  # Wrong prediction
                    self.weights[i] -= self.weight_adaptation_rate * self.weights[i]
        
        # Normalize weights
        self.weights = np.maximum(0.01, self.weights)  # Ensure minimum weight
        self.weights = self.weights / np.sum(self.weights)
    
    def _update_threshold(self, prediction, actual):
        """
        Update dynamic threshold based on prediction accuracy.
        
        Args:
            prediction: Predicted outcome ('P' or 'B')
            actual: Actual outcome ('P' or 'B')
        """
        correct = prediction == actual
        
        if correct:
            # If prediction was correct, slightly decrease threshold
            self.threshold = max(0.5, self.threshold - self.threshold_adaptation_rate * 0.5)
            self.consecutive_losses = 0
        else:
            # If prediction was wrong, increase threshold
            self.threshold = min(0.8, self.threshold + self.threshold_adaptation_rate)
            self.consecutive_losses += 1
        
        # Check if we should enter mean reversion mode
        if self.consecutive_losses >= 3:
            self.mean_reversion_mode = True
            self.mean_reversion_counter = 5  # Stay in mean reversion mode for 5 bets
        
        # Decrease mean reversion counter
        if self.mean_reversion_mode:
            self.mean_reversion_counter -= 1
            if self.mean_reversion_counter <= 0:
                self.mean_reversion_mode = False
    
    def _predict_next_outcome(self, outcomes):
        """
        Predict the next outcome based on momentum indicators.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        # Calculate momentum for each window size
        momentums = []
        for window_size in self.window_sizes:
            if len(outcomes) >= window_size:
                momentum = self._calculate_momentum(outcomes, window_size)
                momentums.append(momentum)
            else:
                momentums.append(0)
        
        # Calculate weighted momentum
        weighted_momentum = np.sum(np.array(momentums) * self.weights)
        
        # Apply banker bias
        weighted_momentum -= self.banker_bias
        
        # If in mean reversion mode, reverse the prediction
        if self.mean_reversion_mode:
            weighted_momentum = -weighted_momentum
        
        # Calculate confidence
        confidence = abs(weighted_momentum)
        
        logger.debug(f"Weighted momentum: {weighted_momentum:.3f}, Confidence: {confidence:.3f}, Threshold: {self.threshold:.3f}")
        
        # Make prediction based on confidence and threshold
        if confidence > self.threshold:
            return 'P' if weighted_momentum > 0 else 'B'
        else:
            # When confidence is low, use recent trend
            recent = outcomes[-10:] if len(outcomes) >= 10 else outcomes
            p_count = recent.count('P')
            b_count = recent.count('B')
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            return 'P' if p_count > b_count else 'B'
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using adaptive momentum analysis.
        
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
        
        # Update memory with new outcome
        if len(filtered) > 0 and (len(self.outcome_memory) == 0 or filtered[-1] != self.outcome_memory[-1]):
            self.outcome_memory.append(filtered[-1])
        
        # Make prediction
        prediction = self._predict_next_outcome(filtered)
        
        # Store prediction
        self.prediction_memory.append(prediction)
        
        # Update weights and threshold if we have the actual outcome
        if len(self.outcome_memory) >= 2 and len(self.prediction_memory) >= 2:
            self._update_weights(self.prediction_memory[-2], self.outcome_memory[-1])
            self._update_threshold(self.prediction_memory[-2], self.outcome_memory[-1])
            
            # Update performance tracking
            correct = self.prediction_memory[-2] == self.outcome_memory[-1]
            if correct:
                self.correct_predictions += 1
            self.total_predictions += 1
        
        # Log performance
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logger.debug(f"Current accuracy: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")
        
        return prediction
