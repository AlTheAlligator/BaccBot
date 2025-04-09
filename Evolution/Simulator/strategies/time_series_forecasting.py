"""
Time Series Forecasting strategy implementation.
"""

import logging
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TimeSeriesForecastingStrategy(BaseStrategy):
    """
    Time Series Forecasting strategy for baccarat betting.
    
    This strategy uses ARIMA (AutoRegressive Integrated Moving Average) models
    to forecast the next outcome based on historical patterns in the time series.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Time Series Forecasting strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available. Using fallback mode for Time Series Forecasting strategy.")
            self.statsmodels_available = False
        else:
            self.statsmodels_available = True
        
        # Strategy parameters
        self.window_size = params.get('window_size', 50)  # Size of the window for analysis
        self.min_samples = params.get('min_samples', 30)  # Minimum samples before making predictions
        self.p = params.get('p', 2)  # AR order
        self.d = params.get('d', 0)  # Differencing order
        self.q = params.get('q', 2)  # MA order
        self.use_auto_arima = params.get('use_auto_arima', True)  # Whether to automatically determine ARIMA parameters
        self.confidence_threshold = params.get('confidence_threshold', 0.55)  # Threshold for making a bet
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        
        # For tracking state
        self.numeric_history = deque(maxlen=self.window_size)
        self.model = None
        self.last_prediction = None
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logger.info(f"Initialized Time Series Forecasting strategy with window_size={self.window_size}")
    
    def _outcomes_to_numeric(self, outcomes):
        """
        Convert outcome strings to numeric values for time series analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            list: Numeric values (1 for P, 0 for B)
        """
        return [1 if o == 'P' else 0 for o in outcomes]
    
    def _update_numeric_history(self, outcomes):
        """
        Update the numeric history with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        numeric = self._outcomes_to_numeric(outcomes)
        self.numeric_history.extend(numeric)
    
    def _determine_arima_params(self, series):
        """
        Automatically determine the best ARIMA parameters.
        
        Args:
            series: Numeric time series data
            
        Returns:
            tuple: (p, d, q) parameters
        """
        if not self.statsmodels_available:
            return self.p, self.d, self.q
        
        try:
            # Check for stationarity
            diff = 0
            diff_series = np.array(series)
            
            # Simple stationarity check - if mean is changing, difference once
            if len(series) >= 20:
                first_half_mean = np.mean(series[:len(series)//2])
                second_half_mean = np.mean(series[len(series)//2:])
                if abs(first_half_mean - second_half_mean) > 0.1:
                    diff = 1
                    diff_series = np.diff(series)
            
            # Calculate ACF and PACF
            acf_values = acf(diff_series, nlags=10, fft=False)
            pacf_values = pacf(diff_series, nlags=10)
            
            # Determine p from PACF
            p = 0
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > 0.2:
                    p = max(p, i)
            
            # Determine q from ACF
            q = 0
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > 0.2:
                    q = max(q, i)
            
            # Limit to reasonable values
            p = min(p, 3)
            q = min(q, 3)
            
            return p, diff, q
        except Exception as e:
            logger.warning(f"Error determining ARIMA parameters: {e}")
            return self.p, self.d, self.q
    
    def _fit_arima_model(self, series):
        """
        Fit an ARIMA model to the time series data.
        
        Args:
            series: Numeric time series data
            
        Returns:
            model: Fitted ARIMA model
        """
        if not self.statsmodels_available or len(series) < self.min_samples:
            return None
        
        try:
            # Determine ARIMA parameters
            if self.use_auto_arima:
                p, d, q = self._determine_arima_params(series)
            else:
                p, d, q = self.p, self.d, self.q
            
            # Fit ARIMA model
            model = ARIMA(series, order=(p, d, q))
            fitted_model = model.fit()
            
            return fitted_model
        except Exception as e:
            logger.warning(f"Error fitting ARIMA model: {e}")
            return None
    
    def _predict_next_outcome(self, model, series):
        """
        Predict the next outcome using the fitted model.
        
        Args:
            model: Fitted ARIMA model
            series: Numeric time series data
            
        Returns:
            tuple: (prediction, confidence)
        """
        if model is None:
            # Fallback to simple frequency analysis
            p_count = sum(series)
            b_count = len(series) - p_count
            total = len(series)
            
            if total == 0:
                return 'B', 0.51  # Default to banker with minimal confidence
            
            p_prob = p_count / total
            b_prob = b_count / total
            
            # Apply banker bias
            b_prob += self.banker_bias
            
            # Normalize
            total_prob = p_prob + b_prob
            p_prob /= total_prob
            b_prob /= total_prob
            
            # Determine prediction and confidence
            if p_prob > b_prob:
                return 'P', p_prob
            else:
                return 'B', b_prob
        
        try:
            # Get forecast
            forecast = model.forecast(steps=1)
            predicted_value = forecast[0]
            
            # Convert to probability
            p_prob = predicted_value
            b_prob = 1 - predicted_value
            
            # Apply banker bias
            b_prob += self.banker_bias
            
            # Normalize
            total_prob = p_prob + b_prob
            p_prob /= total_prob
            b_prob /= total_prob
            
            # Determine prediction and confidence
            if p_prob > b_prob:
                return 'P', p_prob
            else:
                return 'B', b_prob
        except Exception as e:
            logger.warning(f"Error making prediction: {e}")
            return 'B', 0.51  # Default to banker with minimal confidence
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using time series forecasting.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)
        
        # Filter out ties
        filtered_outcomes = [o for o in outcomes if o in ['P', 'B']]
        
        # Not enough data - default to Banker (slight edge due to commission)
        if len(filtered_outcomes) < self.min_samples:
            return 'B'
        
        # Update numeric history
        self._update_numeric_history(filtered_outcomes)
        
        # Get numeric series
        series = list(self.numeric_history)
        
        # Fit ARIMA model
        self.model = self._fit_arima_model(series)
        
        # Make prediction
        bet, confidence = self._predict_next_outcome(self.model, series)
        
        # Store prediction for evaluation
        self.last_prediction = bet
        
        # Always return a bet (no skipping)
        return bet
    
    def evaluate_prediction(self, actual_outcome):
        """
        Evaluate the last prediction against the actual outcome.
        
        Args:
            actual_outcome: The actual outcome ('P', 'B', or 'T')
        """
        if self.last_prediction is None or actual_outcome == 'T':
            return
        
        self.total_predictions += 1
        if actual_outcome == self.last_prediction:
            self.correct_predictions += 1
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        return {
            "correct_predictions": self.correct_predictions,
            "total_predictions": self.total_predictions,
            "accuracy": accuracy,
            "arima_params": (self.p, self.d, self.q) if self.model is None else self.model.model.order
        }
