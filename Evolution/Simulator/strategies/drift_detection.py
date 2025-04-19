"""
Drift Detection Strategy

This strategy detects and exploits temporary statistical drifts in the
baccarat game outcomes using change point detection algorithms.
"""

import numpy as np
from collections import deque
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DriftDetectionStrategy:
    """
    A strategy that detects and exploits temporary statistical drifts in baccarat outcomes.
    
    Features:
    - CUSUM (Cumulative Sum) change point detection
    - Page-Hinkley test for drift detection
    - Adaptive window sizing
    - Drift exploitation with confidence-based betting
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Drift Detection strategy.
        
        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}
        
        # Core parameters
        self.min_window = params.get('min_window', 5)
        self.max_window = params.get('max_window', 12)
        self.drift_threshold = params.get('drift_threshold', 2.0)
        self.reset_threshold = params.get('reset_threshold', 1.0)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.min_samples = params.get('min_samples', 5)
        
        # Advanced parameters
        self.use_cusum = params.get('use_cusum', True)
        self.use_page_hinkley = params.get('use_page_hinkley', True)
        self.cusum_weight = params.get('cusum_weight', 0.6)
        self.ph_weight = params.get('ph_weight', 0.4)
        self.cusum_delta = params.get('cusum_delta', 0.05)
        self.ph_delta = params.get('ph_delta', 0.005)
        self.ph_lambda = params.get('ph_lambda', 0.1)
        self.adaptive_window = params.get('adaptive_window', True)
        self.confidence_multiplier = params.get('confidence_multiplier', 1.5)
        
        # Expected probabilities (theoretical)
        self.expected_p_prob = 0.4462  # Player win probability
        self.expected_b_prob = 0.4585  # Banker win probability
        self.expected_t_prob = 0.0953  # Tie probability
        
        # Normalized for non-tie outcomes
        self.expected_p_norm = self.expected_p_prob / (self.expected_p_prob + self.expected_b_prob)
        self.expected_b_norm = self.expected_b_prob / (self.expected_p_prob + self.expected_b_prob)
        
        # State tracking
        self.numeric_history = []  # 1 for Player, 0 for Banker, -1 for Tie
        self.current_window = self.min_window
        self.drift_detected = False
        self.drift_direction = 0  # 1 for Player drift, -1 for Banker drift
        self.drift_confidence = 0.0
        self.drift_start_index = 0
        
        # CUSUM variables
        self.cusum_p = 0.0
        self.cusum_b = 0.0
        self.cusum_last_reset = 0
        
        # Page-Hinkley variables
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.ph_max = 0.0
        
    def _outcomes_to_numeric(self, outcomes: List[str]) -> List[int]:
        """
        Convert outcome strings to numeric values.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            list: Numeric outcomes (1 for P, 0 for B, -1 for T)
        """
        return [1 if o == 'P' else (0 if o == 'B' else -1) for o in outcomes]
    
    def _calculate_drift_cusum(self, numeric_outcomes: List[int]) -> Tuple[float, float, int]:
        """
        Calculate drift using CUSUM (Cumulative Sum) algorithm.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (drift_magnitude, drift_direction, drift_start_index)
        """
        if len(numeric_outcomes) < self.min_samples:
            return 0.0, 0, 0
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < self.min_samples:
            return 0.0, 0, 0
            
        # Reset CUSUM if it's been a while since last reset
        if len(filtered) - self.cusum_last_reset > self.max_window * 2:
            self.cusum_p = 0.0
            self.cusum_b = 0.0
            self.cusum_last_reset = len(filtered)
        
        # Calculate CUSUM for Player and Banker
        for i in range(max(0, self.cusum_last_reset), len(filtered)):
            # Deviation from expected probability
            deviation = filtered[i] - self.expected_p_norm
            
            # Update CUSUM values
            self.cusum_p = max(0, self.cusum_p + deviation - self.cusum_delta)
            self.cusum_b = max(0, self.cusum_b - deviation - self.cusum_delta)
            
        # Determine drift direction and magnitude
        if self.cusum_p > self.cusum_b:
            drift_direction = 1  # Player drift
            drift_magnitude = self.cusum_p
        else:
            drift_direction = -1  # Banker drift
            drift_magnitude = self.cusum_b
            
        # Estimate drift start index
        drift_start_index = max(0, len(filtered) - int(drift_magnitude / self.cusum_delta))
        
        return drift_magnitude, drift_direction, drift_start_index
    
    def _calculate_drift_page_hinkley(self, numeric_outcomes: List[int]) -> Tuple[float, float, int]:
        """
        Calculate drift using Page-Hinkley test.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (drift_magnitude, drift_direction, drift_start_index)
        """
        if len(numeric_outcomes) < self.min_samples:
            return 0.0, 0, 0
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < self.min_samples:
            return 0.0, 0, 0
            
        # Calculate running mean
        mean_p = np.mean(filtered)
        
        # Deviation from expected probability
        deviation = mean_p - self.expected_p_norm
        
        # Update Page-Hinkley variables
        self.ph_sum += deviation
        self.ph_min = min(self.ph_min, self.ph_sum)
        self.ph_max = max(self.ph_max, self.ph_sum)
        
        # Calculate PH statistics
        ph_up = self.ph_sum - self.ph_min - self.ph_delta
        ph_down = self.ph_max - self.ph_sum - self.ph_delta
        
        # Determine drift direction and magnitude
        if ph_up > ph_down:
            drift_direction = 1  # Player drift
            drift_magnitude = ph_up
        else:
            drift_direction = -1  # Banker drift
            drift_magnitude = ph_down
            
        # Estimate drift start index
        drift_length = int(drift_magnitude / self.ph_lambda)
        drift_start_index = max(0, len(filtered) - drift_length)
        
        return drift_magnitude, drift_direction, drift_start_index
    
    def _detect_drift(self, numeric_outcomes: List[int]) -> Tuple[bool, int, float]:
        """
        Detect statistical drift in the outcome sequence.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (drift_detected, drift_direction, drift_confidence)
        """
        drift_detected = False
        drift_direction = 0
        drift_confidence = 0.0
        drift_start_index = 0
        
        # Calculate drift using CUSUM
        if self.use_cusum:
            cusum_magnitude, cusum_direction, cusum_start = self._calculate_drift_cusum(numeric_outcomes)
            cusum_detected = cusum_magnitude > self.drift_threshold
            
            if cusum_detected:
                drift_detected = True
                drift_direction = cusum_direction
                drift_confidence = min(1.0, cusum_magnitude / (self.drift_threshold * 2))
                drift_start_index = cusum_start
        
        # Calculate drift using Page-Hinkley
        if self.use_page_hinkley:
            ph_magnitude, ph_direction, ph_start = self._calculate_drift_page_hinkley(numeric_outcomes)
            ph_detected = ph_magnitude > self.drift_threshold
            
            if ph_detected:
                # If both methods detect drift, combine them
                if drift_detected:
                    # If they agree on direction, combine confidence
                    if ph_direction == drift_direction:
                        drift_confidence = (drift_confidence * self.cusum_weight + 
                                           min(1.0, ph_magnitude / (self.drift_threshold * 2)) * self.ph_weight)
                        drift_start_index = max(drift_start_index, ph_start)
                    else:
                        # If they disagree, go with the stronger signal
                        if ph_magnitude > cusum_magnitude:
                            drift_direction = ph_direction
                            drift_confidence = min(1.0, ph_magnitude / (self.drift_threshold * 2))
                            drift_start_index = ph_start
                else:
                    drift_detected = True
                    drift_direction = ph_direction
                    drift_confidence = min(1.0, ph_magnitude / (self.drift_threshold * 2))
                    drift_start_index = ph_start
        
        return drift_detected, drift_direction, drift_confidence, drift_start_index
    
    def _adapt_window(self, drift_detected: bool, drift_confidence: float):
        """
        Adapt window size based on drift detection.
        
        Args:
            drift_detected: Whether drift was detected
            drift_confidence: Confidence in the drift detection
        """
        if not self.adaptive_window:
            return
            
        if drift_detected:
            # Decrease window size to focus on recent drift
            self.current_window = max(self.min_window, 
                                     int(self.current_window * (1.0 - drift_confidence * 0.3)))
        else:
            # Gradually increase window size when no drift is detected
            self.current_window = min(self.max_window, self.current_window + 1)
    
    def _calculate_drift_probabilities(self, numeric_outcomes: List[int], 
                                      drift_direction: int, 
                                      drift_start_index: int) -> Dict[str, float]:
        """
        Calculate outcome probabilities based on detected drift.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            drift_direction: Direction of the drift (1 for Player, -1 for Banker)
            drift_start_index: Index where the drift started
            
        Returns:
            dict: Outcome probabilities
        """
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < self.min_samples or drift_start_index >= len(filtered):
            return {'P': self.expected_p_norm, 'B': self.expected_b_norm}
            
        # Get outcomes since drift started
        drift_outcomes = filtered[drift_start_index:]
        
        if len(drift_outcomes) < self.min_samples:
            return {'P': self.expected_p_norm, 'B': self.expected_b_norm}
            
        # Calculate probabilities from drift period
        p_count = sum(1 for o in drift_outcomes if o == 1)
        b_count = len(drift_outcomes) - p_count
        
        p_prob = p_count / len(drift_outcomes)
        b_prob = b_count / len(drift_outcomes)
        
        # Apply banker bias
        b_prob += b_prob * self.banker_bias
        
        # Normalize
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        return {'P': p_prob, 'B': b_prob}
    
    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using drift detection.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Always start from game 7
        if len(outcomes) < 7:
            return 'B'  # Default to Banker for initial games
            
        # Convert outcomes to numeric
        numeric = self._outcomes_to_numeric(outcomes)
        self.numeric_history = numeric
        
        # Not enough data for drift detection
        if len(numeric) < self.min_samples:
            return 'B'  # Default to Banker
        
        # Detect drift
        drift_detected, drift_direction, drift_confidence, drift_start_index = self._detect_drift(numeric)
        
        # Update state
        self.drift_detected = drift_detected
        self.drift_direction = drift_direction
        self.drift_confidence = drift_confidence
        self.drift_start_index = drift_start_index
        
        # Adapt window size
        self._adapt_window(drift_detected, drift_confidence)
        
        # If no significant drift is detected, use default strategy
        if not drift_detected or drift_confidence < 0.2:
            return 'B'  # Default to Banker
        
        # Calculate probabilities based on drift
        probs = self._calculate_drift_probabilities(numeric, drift_direction, drift_start_index)
        
        # Enhance the signal based on confidence
        if drift_direction == 1:  # Player drift
            probs['P'] = probs['P'] * (1 + drift_confidence * self.confidence_multiplier)
        else:  # Banker drift
            probs['B'] = probs['B'] * (1 + drift_confidence * self.confidence_multiplier)
            
        # Normalize
        total = probs['P'] + probs['B']
        probs['P'] /= total
        probs['B'] /= total
        
        # Make decision
        return 'P' if probs['P'] > probs['B'] else 'B'
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.
        
        Returns:
            dict: Strategy statistics
        """
        # Filter out ties
        filtered = [o for o in self.numeric_history if o != -1]
        
        # Calculate recent player frequency
        recent_window = min(self.current_window, len(filtered))
        recent = filtered[-recent_window:] if filtered else []
        
        p_freq = sum(1 for o in recent if o == 1) / len(recent) if recent else 0
        
        # Calculate drift statistics
        drift_stats = {
            "detected": self.drift_detected,
            "direction": "Player" if self.drift_direction == 1 else "Banker" if self.drift_direction == -1 else "None",
            "confidence": f"{self.drift_confidence:.2f}",
            "length": len(filtered) - self.drift_start_index if self.drift_detected else 0
        }
        
        # CUSUM statistics
        cusum_stats = {
            "cusum_p": f"{self.cusum_p:.2f}",
            "cusum_b": f"{self.cusum_b:.2f}",
            "threshold": self.drift_threshold
        }
        
        # Page-Hinkley statistics
        ph_stats = {
            "ph_up": f"{self.ph_sum - self.ph_min - self.ph_delta:.2f}",
            "ph_down": f"{self.ph_max - self.ph_sum - self.ph_delta:.2f}",
            "threshold": self.drift_threshold
        }
        
        return {
            "strategy": "Drift Detection",
            "current_window": self.current_window,
            "player_frequency": f"{p_freq:.2f}",
            "expected_p_freq": f"{self.expected_p_norm:.2f}",
            "drift": drift_stats,
            "cusum": cusum_stats,
            "page_hinkley": ph_stats
        }
