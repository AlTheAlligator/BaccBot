import logging
import numpy as np
from collections import deque

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MomentumOscillatorStrategy(BaseStrategy):
    """
    A strategy that uses a technical analysis inspired oscillator to detect momentum shifts.
    
    This approach tracks the "momentum" of player vs banker outcomes and uses oscillator
    techniques to identify potential reversal points or continuation opportunities.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Momentum Oscillator strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.short_window = params.get('short_window', 8)  # Fast moving average window
        self.long_window = params.get('long_window', 17)  # Slow moving average window
        self.min_samples = params.get('min_samples', 3)  # Minimum samples before making predictions
        self.overbought_threshold = params.get('overbought_threshold', 42)  # Threshold for overbought condition (0-100)
        self.oversold_threshold = params.get('oversold_threshold', 70)  # Threshold for oversold condition (0-100)
        self.signal_line_period = params.get('signal_line_period', 5)  # Period for signal line calculation
        self.reversal_weight = params.get('reversal_weight', 3.2687755102040814)  # Weight for reversal signals (0-1)
        self.trend_weight = params.get('trend_weight', 4.59265306122449)  # Weight for trend continuation signals (0-1)
        self.confidence_threshold = params.get('confidence_threshold', 0.29591836734693877)  # Min confidence to place bet
        self.banker_bias = params.get('banker_bias', 0.1959387755102041)  # Slight bias towards banker bets
        self.use_stochastic = params.get('use_stochastic', False)  # Whether to use stochastic oscillator in addition to RSI
        
        # Initialize oscillator values
        self.rsi_values = deque(maxlen=50)  # Recent RSI values
        self.stoch_values = deque(maxlen=50)  # Recent Stochastic values
        self.signal_line = deque(maxlen=50)  # Signal line values
        
        # History of outcomes converted to numerical values (P=1, B=-1)
        self.numeric_outcomes = deque(maxlen=max(100, self.long_window * 2))
        
        # For tracking performance
        self.signal_history = deque(maxlen=100)  # Recent signals generated
        
        logger.info(f"Initialized Momentum Oscillator strategy with short_window={self.short_window}, "
                   f"long_window={self.long_window}, overbought={self.overbought_threshold}, "
                   f"oversold={self.oversold_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on oscillator analysis.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for oscillator calculation
            logger.debug(f"Not enough data for oscillator ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update numeric history
        self._update_numeric_history(non_tie_outcomes)
        
        # Calculate oscillators
        self._calculate_oscillators()
        
        # Get signals from oscillators
        p_prob, b_prob, signal_strength = self._get_signals()
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize probabilities
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        logger.debug(f"Oscillator prediction: P={p_prob:.3f}, B={b_prob:.3f}, strength={signal_strength:.3f}")
        
        # Make decision based on signal strength and probabilities
        if signal_strength >= self.confidence_threshold:
            if p_prob > b_prob:
                return "P"
            else:
                return "B"
        else:
            logger.debug(f"Signal strength too low: {signal_strength:.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _update_numeric_history(self, outcomes):
        """
        Convert outcome strings to numeric values for oscillator calculation.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Convert outcomes to numeric values (P=1, B=-1)
        numeric = []
        for outcome in outcomes:
            if outcome == 'P':
                numeric.append(1)
            elif outcome == 'B':
                numeric.append(-1)
        
        # Update history
        self.numeric_outcomes.extend(numeric)
    
    def _calculate_oscillators(self):
        """Calculate RSI and optionally Stochastic oscillator values."""
        # Calculate RSI
        self.rsi_values.append(self._calculate_rsi())
        
        # Calculate Stochastic if enabled
        if self.use_stochastic:
            self.stoch_values.append(self._calculate_stochastic())
        
        # Calculate signal line (moving average of oscillator)
        if len(self.rsi_values) >= self.signal_line_period:
            signal = sum(list(self.rsi_values)[-self.signal_line_period:]) / self.signal_line_period
            self.signal_line.append(signal)
    
    def _calculate_rsi(self):
        """
        Calculate RSI (Relative Strength Index) value.
        
        Returns:
            float: RSI value (0-100)
        """
        if len(self.numeric_outcomes) < self.short_window:
            return 50  # Default neutral value
        
        # Get recent values for calculation
        values = list(self.numeric_outcomes)[-self.short_window:]
        gains = []
        losses = []
        
        # Calculate gains and losses
        for i in range(1, len(values)):
            change = values[i] - values[i-1]
            if change > 0:
                gains.append(abs(change))
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Calculate RSI
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self):
        """
        Calculate Stochastic oscillator value.
        
        Returns:
            float: Stochastic value (0-100)
        """
        if len(self.numeric_outcomes) < self.long_window:
            return 50  # Default neutral value
        
        # Get values for calculation
        values = list(self.numeric_outcomes)[-self.long_window:]
        
        # Find highest high and lowest low
        highest = max(values)
        lowest = min(values)
        
        # Get closing value (most recent)
        close = values[-1]
        
        # Calculate stochastic
        if highest == lowest:
            return 50  # Neutral when no range
        else:
            stoch = ((close - lowest) / (highest - lowest)) * 100
        
        return stoch
    
    def _get_signals(self):
        """
        Get trading signals from oscillator values.
        
        Returns:
            tuple: (p_prob, b_prob, signal_strength)
        """
        if not self.rsi_values:
            return 0.5, 0.5, 0
        
        # Get current RSI
        current_rsi = self.rsi_values[-1]
        
        # Initialize base probabilities
        p_prob = 0.5
        b_prob = 0.5
        signal_strength = 0
        
        # Calculate trend signals
        trend_signal = 0
        if len(self.rsi_values) >= 2:
            rsi_change = self.rsi_values[-1] - self.rsi_values[-2]
            trend_signal = rsi_change / 100  # Normalize to -1 to 1 range
        
        # Calculate reversal signals
        reversal_signal = 0
        if current_rsi > self.overbought_threshold:
            # Overbought - favor Banker
            reversal_signal = (current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
            reversal_signal = -reversal_signal  # Negative for reversal
        elif current_rsi < self.oversold_threshold:
            # Oversold - favor Player
            reversal_signal = (self.oversold_threshold - current_rsi) / self.oversold_threshold
        
        # Add stochastic confirmation if enabled
        stoch_signal = 0
        if self.use_stochastic and self.stoch_values:
            current_stoch = self.stoch_values[-1]
            if current_stoch > 80:
                stoch_signal = -0.5  # Overbought confirmation
            elif current_stoch < 20:
                stoch_signal = 0.5  # Oversold confirmation
        
        # Combine signals with weights
        combined_signal = (
            trend_signal * self.trend_weight +
            reversal_signal * self.reversal_weight +
            stoch_signal * 0.5  # Stochastic has lower weight
        )
        
        # Convert combined signal to probabilities
        if combined_signal > 0:
            # Favor Player
            shift = min(0.4, abs(combined_signal))
            p_prob = 0.5 + shift
            b_prob = 0.5 - shift
        else:
            # Favor Banker
            shift = min(0.4, abs(combined_signal))
            p_prob = 0.5 - shift
            b_prob = 0.5 + shift
        
        # Calculate signal strength (confidence)
        signal_strength = min(1.0, abs(combined_signal) * 1.5)
        
        return p_prob, b_prob, signal_strength
    
    def get_oscillator_values(self):
        """Get current oscillator values for debugging."""
        return {
            "rsi": list(self.rsi_values),
            "stochastic": list(self.stoch_values) if self.use_stochastic else None,
            "signal_line": list(self.signal_line),
            "latest_values": {
                "rsi": self.rsi_values[-1] if self.rsi_values else None,
                "stochastic": self.stoch_values[-1] if self.stoch_values else None,
                "signal": self.signal_line[-1] if self.signal_line else None
            }
        }