"""
Fractal Analysis strategy implementation.
"""

import logging
import numpy as np
from collections import defaultdict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class FractalAnalysisStrategy(BaseStrategy):
    """
    Fractal Analysis strategy that looks for self-similar patterns at different scales.

    This strategy applies fractal analysis techniques like Hurst exponent calculation
    and fractal dimension estimation to identify self-similarity and long-memory
    properties in the outcome sequence.
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Fractal Analysis strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)

        # Strategy parameters
        self.min_scale = params.get('min_scale', 5)
        self.max_scale = params.get('max_scale', 50)
        self.hurst_threshold = params.get('hurst_threshold', 0.6)
        self.min_samples = params.get('min_samples', 50)  # Minimum samples for fractal analysis
        self.banker_bias = params.get('banker_bias', 0.01)

        # Initialize tracking variables
        self.numeric_history = []  # 1 for Player, 0 for Banker
        self.hurst_exponent = 0.5  # Default to random walk
        self.fractal_dimension = 2.0  # Default to random walk

        logger.info(f"Initialized Fractal Analysis strategy with hurst_threshold={self.hurst_threshold}")

    def _outcomes_to_numeric(self, outcomes):
        """
        Convert outcome strings to numeric values for analysis.

        Args:
            outcomes: List of outcomes ('P', 'B')

        Returns:
            list: Numeric values (1 for P, 0 for B)
        """
        return [1 if o == 'P' else 0 for o in outcomes]

    def _calculate_hurst_exponent(self, time_series):
        """
        Calculate Hurst exponent using R/S analysis.

        Args:
            time_series: Numeric time series data

        Returns:
            float: Hurst exponent
        """
        if len(time_series) < self.min_scale:
            return 0.5  # Default to random walk for insufficient data

        # Convert to numpy array
        series = np.array(time_series)

        # Calculate returns (differences)
        returns = np.diff(series)

        # Determine scales to use
        max_scale = min(self.max_scale, len(returns) // 2)
        scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale), num=10).astype(int)
        scales = np.unique(scales)  # Remove duplicates

        # Calculate R/S values for each scale
        rs_values = []
        for scale in scales:
            # Skip if scale is too large
            if scale >= len(returns):
                continue

            # Split returns into windows of size 'scale'
            num_windows = len(returns) // scale
            if num_windows == 0:
                continue

            # Calculate R/S for each window
            rs_scale = []
            for i in range(num_windows):
                window = returns[i*scale:(i+1)*scale]

                # Calculate mean-adjusted series
                mean_adj = window - np.mean(window)

                # Calculate cumulative deviation
                cum_dev = np.cumsum(mean_adj)

                # Calculate range (max - min of cumulative deviation)
                r = np.max(cum_dev) - np.min(cum_dev)

                # Calculate standard deviation
                s = np.std(window)

                # Avoid division by zero
                if s > 0:
                    rs_scale.append(r / s)

            # Calculate average R/S for this scale
            if rs_scale:
                rs_values.append(np.mean(rs_scale))

        # If not enough valid scales, return default
        if len(scales) < 4 or len(rs_values) < 4:
            return 0.5

        # Log-log regression to estimate Hurst exponent
        log_scales = np.log10(scales[:len(rs_values)])
        log_rs = np.log10(rs_values)

        # Linear regression with warning suppression
        try:
            # Suppress the RankWarning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                hurst, _ = np.polyfit(log_scales, log_rs, 1)
        except Exception as e:
            logger.debug(f"Error in Hurst exponent polyfit: {e}")
            # Fallback to a simple linear regression
            if len(log_rs) >= 2:
                hurst = (log_rs[-1] - log_rs[0]) / (log_scales[-1] - log_scales[0])
            else:
                hurst = 0.5

        return hurst

    def _calculate_fractal_dimension(self, time_series):
        """
        Estimate fractal dimension using box-counting method.

        Args:
            time_series: Numeric time series data

        Returns:
            float: Fractal dimension
        """
        if len(time_series) < self.min_scale:
            return 2.0  # Default to random walk for insufficient data

        # Convert to numpy array
        series = np.array(time_series)

        # Determine scales to use
        max_scale = min(self.max_scale, len(series) // 4)
        scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale), num=10).astype(int)
        scales = np.unique(scales)  # Remove duplicates

        # Calculate box counts for each scale
        box_counts = []
        for scale in scales:
            # Skip if scale is too large
            if scale >= len(series):
                continue

            # Count boxes needed to cover the series
            boxes = set()
            for i in range(len(series) - scale + 1):
                # Use min and max as box coordinates
                box = (int(np.min(series[i:i+scale]) / scale),
                       int(np.max(series[i:i+scale]) / scale))
                boxes.add(box)

            box_counts.append(len(boxes))

        # If not enough valid scales, return default
        if len(scales) < 4 or len(box_counts) < 4:
            return 2.0

        # Log-log regression to estimate fractal dimension
        log_scales = np.log10(scales[:len(box_counts)])
        log_counts = np.log10(box_counts)

        # Linear regression with warning suppression
        try:
            # Suppress the RankWarning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                slope, _ = np.polyfit(log_scales, log_counts, 1)
        except Exception as e:
            logger.debug(f"Error in fractal dimension polyfit: {e}")
            # Fallback to a simple linear regression
            if len(log_counts) >= 2:
                slope = (log_counts[-1] - log_counts[0]) / (log_scales[-1] - log_scales[0])
            else:
                slope = -2.0

        # Fractal dimension is the negative of the slope
        fractal_dim = -slope

        return fractal_dim

    def _detect_trend(self, outcomes):
        """
        Detect trend in recent outcomes.

        Args:
            outcomes: List of outcomes ('P', 'B')

        Returns:
            str: Detected trend ('P' or 'B')
        """
        if len(outcomes) < 10:
            return 'B'  # Default to Banker

        # Use recent outcomes for trend detection
        recent = outcomes[-10:]
        p_count = recent.count('P')
        b_count = recent.count('B')

        # Check if there's a clear trend
        if p_count > b_count * 1.5:
            return 'P'  # Strong Player trend
        elif b_count > p_count * 1.5:
            return 'B'  # Strong Banker trend

        # Check for acceleration
        first_half = outcomes[-10:-5]
        second_half = outcomes[-5:]

        p_first = first_half.count('P')
        p_second = second_half.count('P')

        # If Player outcomes are accelerating
        if p_second > p_first:
            return 'P'
        else:
            return 'B'

    def _calculate_detrended_fluctuation(self, time_series):
        """
        Calculate detrended fluctuation analysis (DFA) exponent.

        Args:
            time_series: Numeric time series data

        Returns:
            float: DFA exponent
        """
        if len(time_series) < self.min_scale:
            return 0.5  # Default to random walk for insufficient data

        # Convert to numpy array
        series = np.array(time_series)

        # Calculate cumulative sum
        profile = np.cumsum(series - np.mean(series))

        # Determine scales to use
        max_scale = min(self.max_scale, len(profile) // 4)
        scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale), num=10).astype(int)
        scales = np.unique(scales)  # Remove duplicates

        # Calculate fluctuation for each scale
        fluctuations = []
        for scale in scales:
            # Skip if scale is too large
            if scale >= len(profile):
                continue

            # Number of windows
            num_windows = len(profile) // scale

            # Calculate fluctuation for each window
            fluct = 0
            for i in range(num_windows):
                # Extract window
                window = profile[i*scale:(i+1)*scale]

                # Fit polynomial (linear detrending)
                x = np.arange(len(window))
                try:
                    # Suppress the RankWarning
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', np.RankWarning)
                        coeffs = np.polyfit(x, window, 1)
                    trend = np.polyval(coeffs, x)
                except Exception as e:
                    # Fallback to simple mean if polyfit fails
                    logger.debug(f"Error in window detrending polyfit: {e}")
                    trend = np.ones_like(window) * np.mean(window)

                # Calculate fluctuation (root mean square)
                fluct += np.sum((window - trend) ** 2)

            # Average fluctuation for this scale
            if num_windows > 0:
                fluct = np.sqrt(fluct / (num_windows * scale))
                fluctuations.append(fluct)

        # If not enough valid scales, return default
        if len(scales) < 4 or len(fluctuations) < 4:
            return 0.5

        # Log-log regression to estimate DFA exponent
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluct = np.log10(fluctuations)

        # Linear regression with warning suppression
        try:
            # Suppress the RankWarning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                alpha, _ = np.polyfit(log_scales, log_fluct, 1)
        except Exception as e:
            logger.debug(f"Error in DFA polyfit: {e}")
            # Fallback to a simple linear regression
            if len(log_fluct) >= 2:
                alpha = (log_fluct[-1] - log_fluct[0]) / (log_scales[-1] - log_scales[0])
            else:
                alpha = 0.5

        return alpha

    def get_bet(self, outcomes):
        """
        Determine the next bet using fractal analysis.

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

        # Not enough data for fractal analysis - use simple frequency analysis
        if len(filtered) < self.min_samples:
            p_count = filtered.count('P')
            b_count = filtered.count('B')

            # Apply banker bias
            b_count += b_count * self.banker_bias

            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'

        # Convert to numeric for fractal analysis
        numeric = self._outcomes_to_numeric(filtered)
        self.numeric_history = numeric

        # Calculate Hurst exponent
        self.hurst_exponent = self._calculate_hurst_exponent(numeric)

        # Calculate fractal dimension
        self.fractal_dimension = self._calculate_fractal_dimension(numeric)

        # Calculate DFA exponent
        dfa_exponent = self._calculate_detrended_fluctuation(numeric)

        logger.debug(f"Hurst: {self.hurst_exponent:.3f}, Fractal Dim: {self.fractal_dimension:.3f}, DFA: {dfa_exponent:.3f}")

        # Make decision based on fractal properties
        if self.hurst_exponent > self.hurst_threshold:
            # Persistent series (trend-following)
            trend = self._detect_trend(filtered)
            logger.debug(f"Persistent series (H > {self.hurst_threshold}): following trend {trend}")
            return trend
        elif self.hurst_exponent < 0.5:
            # Anti-persistent series (mean-reverting)
            last_outcome = filtered[-1]
            logger.debug(f"Anti-persistent series (H < 0.5): reversing last outcome {last_outcome}")
            return 'B' if last_outcome == 'P' else 'P'  # Bet opposite of last outcome
        else:
            # Random walk or weak persistence - use frequency analysis
            recent = filtered[-20:]  # Use recent outcomes
            p_count = recent.count('P')
            b_count = recent.count('B')

            # Apply banker bias
            b_count += b_count * self.banker_bias

            logger.debug(f"Random walk (H ≈ 0.5): frequency analysis P={p_count}, B={b_count}")

            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'

    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "hurst_exponent": self.hurst_exponent,
            "fractal_dimension": self.fractal_dimension,
            "hurst_threshold": self.hurst_threshold,
            "history_length": len(self.numeric_history)
        }
