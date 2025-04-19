import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class VolatilityAdaptiveStrategy:
    """
    A strategy that adapts its approach based on detected volatility in the game outcomes.
    
    This strategy measures the "volatility" (rate of change) in game outcomes and
    adjusts its betting strategy accordingly - using a more conservative approach in
    high-volatility periods and a more aggressive approach in stable periods.
    """
    
    def __init__(self, params=None):
        """
        Initialize the Volatility Adaptive strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        params = params or {}
        # Default parameters
        self.short_window = params.get('short_window', 5)  # Window for short-term volatility
        self.medium_window = params.get('medium_window', 10)  # Window for medium-term volatility
        self.long_window = params.get('long_window', 56)  # Window for long-term volatility
        self.min_samples = params.get('min_samples', 6)  # Minimum samples before making predictions
        self.high_volatility_threshold = params.get('high_volatility_threshold', 0.5842105263157895)  # Threshold for high volatility
        self.low_volatility_threshold = params.get('low_volatility_threshold', 0.22631578947368422)  # Threshold for low volatility
        self.confidence_threshold_base = params.get('confidence_threshold_base', 0.6105263157894738)  # Base confidence threshold
        self.confidence_scaling = params.get('confidence_scaling', 0.5)  # How much to adjust confidence by volatility
        self.banker_bias = params.get('banker_bias', 0.016938775510204084)  # Slight bias towards banker bets
        self.use_adaptive_window = params.get('use_adaptive_window', False)  # Whether to adjust analysis window size
        self.statistical_mode = params.get('statistical_mode', 'combined')  # Statistical method for prediction
        self.pattern_length = params.get('pattern_length', 5)  # Length of pattern to consider in pattern mode
        self.min_pattern_occurrences = params.get('min_pattern_occurrences', 3)  # Min occurrences for valid pattern
        
        # Tracking volatility history
        self.volatility_history = deque(maxlen=100)
        self.current_volatility = 0.5  # Default mid-range volatility
        self.last_confidence = 0  # Last confidence level
        
        # Tracking state
        self.outcome_changes = deque(maxlen=100)  # 1 for change, 0 for continuation
        self.streaks = deque(maxlen=50)  # Length of observed streaks
        
        # Analysis mode - changes based on volatility
        self.analysis_mode = "normal"  # normal, conservative, aggressive
        
        logger.info(f"Initialized Volatility Adaptive strategy with volatility thresholds: "
                   f"high={self.high_volatility_threshold}, low={self.low_volatility_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on volatility-adjusted analysis.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable analysis
            logger.debug(f"Not enough data for volatility analysis ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for volatility analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update volatility metrics
        self._update_volatility_metrics(non_tie_outcomes)
        
        # Adjust strategy based on volatility
        self._adapt_to_volatility()
        
        # Get prediction based on current mode and volatility
        p_prob, b_prob = self._get_prediction_probs(non_tie_outcomes)
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize probabilities
        total = p_prob + b_prob
        if total > 0:
            p_prob /= total
            b_prob /= total
        else:
            p_prob = b_prob = 0.5
        
        # Get confidence threshold adjusted for current volatility
        adjusted_threshold = self._get_adjusted_confidence_threshold()
        
        logger.debug(f"Prediction probs: P={p_prob:.3f}, B={b_prob:.3f}, "
                    f"volatility={self.current_volatility:.2f}, mode={self.analysis_mode}, "
                    f"threshold={adjusted_threshold:.3f}")
        
        self.last_confidence = p_prob if p_prob > b_prob else b_prob
        # Make decision based on probabilities and adjusted threshold
        if p_prob > b_prob and p_prob > adjusted_threshold:
            return "P"
        elif b_prob > p_prob and b_prob > adjusted_threshold:
            return "B"
        else:
            logger.debug(f"No clear advantage or below threshold")
            return "SKIP"
    
    def _update_volatility_metrics(self, outcomes):
        """
        Update volatility metrics based on recent outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Calculate streak changes
        self.outcome_changes.clear()
        current_streak = 1
        
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                # Continuation
                current_streak += 1
                self.outcome_changes.append(0)
            else:
                # Change
                self.streaks.append(current_streak)
                current_streak = 1
                self.outcome_changes.append(1)
        
        # Add final streak if it ended at the end of the sequence
        if current_streak > 1:
            self.streaks.append(current_streak)
        
        # Calculate short-term volatility (rate of change)
        short_changes = self._calculate_change_rate(self.short_window, outcomes)
        medium_changes = self._calculate_change_rate(self.medium_window, outcomes)
        long_changes = self._calculate_change_rate(self.long_window, outcomes)
        
        # Weighted average of different time scales
        self.current_volatility = 0.5 * short_changes + 0.3 * medium_changes + 0.2 * long_changes
        
        # Keep track of volatility history
        self.volatility_history.append(self.current_volatility)
    
    def _calculate_change_rate(self, window, outcomes):
        """
        Calculate the rate of change in a given window.
        
        Args:
            window: Window size to analyze
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            float: Change rate (0-1)
        """
        if len(outcomes) < window + 1:
            return 0.5  # Default mid-range if not enough data
            
        recent = outcomes[-window:]
        changes = 0
        
        for i in range(1, len(recent)):
            if recent[i] != recent[i-1]:
                changes += 1
                
        return changes / (len(recent) - 1)
    
    def _adapt_to_volatility(self):
        """
        Adapt strategy parameters based on current volatility.
        """
        # Determine analysis mode based on volatility
        if self.current_volatility >= self.high_volatility_threshold:
            self.analysis_mode = "conservative"
        elif self.current_volatility <= self.low_volatility_threshold:
            self.analysis_mode = "aggressive"
        else:
            self.analysis_mode = "normal"
            
        # Adjust analysis window if adaptive window is enabled
        if self.use_adaptive_window:
            if self.analysis_mode == "conservative":
                # In high volatility, use longer windows for stability
                self.effective_window = int(self.medium_window * 1.5)
            elif self.analysis_mode == "aggressive":
                # In low volatility, use shorter windows for responsiveness
                self.effective_window = int(self.medium_window * 0.7)
            else:
                # Normal mode
                self.effective_window = self.medium_window
        else:
            self.effective_window = self.medium_window
    
    def _get_adjusted_confidence_threshold(self):
        """
        Get confidence threshold adjusted for current volatility.
        
        Returns:
            float: Adjusted confidence threshold
        """
        # High volatility -> higher threshold (more selective)
        # Low volatility -> lower threshold (more opportunities)
        volatility_factor = (self.current_volatility - 0.5) * 2  # Transform to -1 to 1 range
        
        adjustment = volatility_factor * self.confidence_scaling
        
        # Calculate adjusted threshold
        adjusted_threshold = self.confidence_threshold_base + adjustment
        
        # Ensure it stays in reasonable range
        return max(0.51, min(0.75, adjusted_threshold))
    
    def _get_prediction_probs(self, outcomes):
        """
        Calculate prediction probabilities based on current analysis mode.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        # Select analysis method based on mode and statistical approach
        if self.statistical_mode == 'frequency':
            return self._analyze_frequencies(outcomes)
        elif self.statistical_mode == 'pattern':
            return self._analyze_patterns(outcomes)
        elif self.statistical_mode == 'streak':
            return self._analyze_streaks(outcomes)
        elif self.statistical_mode == 'combined':
            # Combine multiple methods
            p1, b1 = self._analyze_frequencies(outcomes)
            p2, b2 = self._analyze_patterns(outcomes)
            p3, b3 = self._analyze_streaks(outcomes)
            
            # Weighted average based on current volatility
            if self.analysis_mode == "conservative":
                weights = (0.5, 0.3, 0.2)  # Favor frequency in high volatility
            elif self.analysis_mode == "aggressive":
                weights = (0.2, 0.5, 0.3)  # Favor patterns in low volatility
            else:
                weights = (0.4, 0.4, 0.2)  # Balanced in normal volatility
                
            p_prob = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
            b_prob = weights[0] * b1 + weights[1] * b2 + weights[2] * b3
            
            return p_prob, b_prob
        else:
            # Default to frequency analysis
            return self._analyze_frequencies(outcomes)
    
    def _analyze_frequencies(self, outcomes):
        """
        Analyze outcome frequencies to predict the next outcome.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        # Use effective window based on current volatility
        window = min(len(outcomes), self.effective_window)
        recent = outcomes[-window:]
        
        # Count occurrences
        p_count = recent.count('P')
        b_count = recent.count('B')
        total = p_count + b_count
        
        # Calculate probabilities
        if total > 0:
            p_prob = p_count / total
            b_prob = b_count / total
        else:
            p_prob = b_prob = 0.5
            
        # In conservative mode (high volatility), fade the current trend slightly
        if self.analysis_mode == "conservative":
            if p_prob > 0.5:
                p_prob = 0.5 + (p_prob - 0.5) * 0.8
                b_prob = 1 - p_prob
            elif b_prob > 0.5:
                b_prob = 0.5 + (b_prob - 0.5) * 0.8
                p_prob = 1 - b_prob
                
        # In aggressive mode (low volatility), amplify the current trend slightly
        elif self.analysis_mode == "aggressive":
            if p_prob > 0.5:
                p_prob = 0.5 + (p_prob - 0.5) * 1.2
                p_prob = min(p_prob, 0.95)  # Cap at 95%
                b_prob = 1 - p_prob
            elif b_prob > 0.5:
                b_prob = 0.5 + (b_prob - 0.5) * 1.2
                b_prob = min(b_prob, 0.95)  # Cap at 95%
                p_prob = 1 - b_prob
        
        return p_prob, b_prob
    
    def _analyze_patterns(self, outcomes):
        """
        Analyze patterns to predict the next outcome.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        # Adjust pattern length based on volatility
        if self.analysis_mode == "conservative":
            # Use longer patterns in high volatility for more certainty
            pattern_length = min(self.pattern_length + 1, 5)
        elif self.analysis_mode == "aggressive":
            # Use shorter patterns in low volatility for more signals
            pattern_length = max(self.pattern_length - 1, 2)
        else:
            pattern_length = self.pattern_length
            
        # Ensure we have enough outcomes for the pattern
        if len(outcomes) < pattern_length + 1:
            return 0.5, 0.5
            
        # Extract current pattern
        current_pattern = tuple(outcomes[-pattern_length:])
        
        # Count pattern occurrences and following outcomes
        p_follow = 0
        b_follow = 0
        
        for i in range(len(outcomes) - pattern_length):
            pattern = tuple(outcomes[i:i+pattern_length])
            if pattern == current_pattern and i + pattern_length < len(outcomes):
                next_outcome = outcomes[i + pattern_length]
                if next_outcome == 'P':
                    p_follow += 1
                elif next_outcome == 'B':
                    b_follow += 1
        
        total_follow = p_follow + b_follow
        
        # Calculate probabilities if pattern occurred enough times
        if total_follow >= self.min_pattern_occurrences:
            p_prob = p_follow / total_follow
            b_prob = b_follow / total_follow
            
            # Apply mode-specific adjustments
            if self.analysis_mode == "conservative":
                # Moderate the prediction in high volatility
                p_prob = 0.5 + (p_prob - 0.5) * 0.8
                b_prob = 0.5 + (b_prob - 0.5) * 0.8
            elif self.analysis_mode == "aggressive":
                # Amplify the prediction in low volatility
                p_prob = 0.5 + (p_prob - 0.5) * 1.2
                b_prob = 0.5 + (b_prob - 0.5) * 1.2
                
            # Normalize
            total = p_prob + b_prob
            if total > 0:
                p_prob /= total
                b_prob /= total
                
            return p_prob, b_prob
        else:
            # Not enough pattern occurrences
            return 0.5, 0.5
    
    def _analyze_streaks(self, outcomes):
        """
        Analyze streak behavior to predict the next outcome.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        if len(outcomes) < 5:  # Need some history to analyze streaks
            return 0.5, 0.5
            
        # Look at recent streaks
        avg_streak_length = sum(self.streaks) / len(self.streaks) if self.streaks else 1
        
        # Count the current streak
        current_streak = 1
        for i in range(len(outcomes) - 2, -1, -1):
            if outcomes[i] == outcomes[-1]:
                current_streak += 1
            else:
                break
        
        # Calculate streak-based probabilities
        if current_streak >= avg_streak_length * 1.5:
            # Well above average streak - likely to break soon
            if outcomes[-1] == 'P':
                return 0.3, 0.7  # Favor B as streak breaker
            else:
                return 0.7, 0.3  # Favor P as streak breaker
        elif current_streak <= avg_streak_length * 0.5 and current_streak > 1:
            # Short streak - likely to continue a bit more
            if outcomes[-1] == 'P':
                return 0.65, 0.35  # Favor P to continue
            else:
                return 0.35, 0.65  # Favor B to continue
        else:
            # Average streak length - neutral
            return 0.5, 0.5
    
    def get_volatility_stats(self):
        """Get volatility statistics for debugging."""
        return {
            "current_volatility": self.current_volatility,
            "analysis_mode": self.analysis_mode,
            "volatility_history": list(self.volatility_history),
            "effective_window": self.effective_window,
            "avg_streak_length": sum(self.streaks) / len(self.streaks) if self.streaks else 0
        }