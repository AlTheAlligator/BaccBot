"""
Multi-Scale Momentum Strategy

This strategy analyzes momentum across different time scales to identify
trends and reversals in baccarat outcomes.
"""

import numpy as np
from collections import deque
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MultiScaleMomentumStrategy:
    """
    A strategy that analyzes momentum across different time scales to identify
    trends and reversals in baccarat outcomes.
    
    Features:
    - Multi-timeframe momentum indicators
    - Oscillator divergence detection
    - Trend strength measurement
    - Overbought/oversold detection
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Multi-Scale Momentum strategy.
        
        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}
        
        # Core parameters
        self.short_window = params.get('short_window', 3)
        self.medium_window = params.get('medium_window', 8)
        self.long_window = params.get('long_window', 12)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.min_samples = params.get('min_samples', 5)
        
        # Advanced parameters
        self.overbought_threshold = params.get('overbought_threshold', 0.7)
        self.oversold_threshold = params.get('oversold_threshold', 0.3)
        self.trend_threshold = params.get('trend_threshold', 0.1)
        self.divergence_threshold = params.get('divergence_threshold', 0.15)
        self.use_rsi = params.get('use_rsi', True)
        self.use_macd = params.get('use_macd', True)
        self.use_stochastic = params.get('use_stochastic', True)
        self.rsi_weight = params.get('rsi_weight', 0.3)
        self.macd_weight = params.get('macd_weight', 0.4)
        self.stochastic_weight = params.get('stochastic_weight', 0.3)
        
        # State tracking
        self.numeric_history = []  # 1 for Player, 0 for Banker, -1 for Tie
        self.momentum_indicators = {}
        self.current_trend = 0  # 1 for Player trend, -1 for Banker trend, 0 for no trend
        self.trend_strength = 0.0
        
    def _outcomes_to_numeric(self, outcomes: List[str]) -> List[int]:
        """
        Convert outcome strings to numeric values.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            list: Numeric outcomes (1 for P, 0 for B, -1 for T)
        """
        return [1 if o == 'P' else (0 if o == 'B' else -1) for o in outcomes]
    
    def _calculate_player_frequency(self, numeric_outcomes: List[int], window: int) -> float:
        """
        Calculate player frequency in a given window.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            window: Size of the window
            
        Returns:
            float: Player frequency (0-1)
        """
        if len(numeric_outcomes) < window:
            return 0.5
            
        # Get the most recent outcomes in the window
        recent = numeric_outcomes[-window:]
        
        # Filter out ties
        filtered = [o for o in recent if o != -1]
        
        if not filtered:
            return 0.5
            
        # Calculate player frequency
        return sum(1 for o in filtered if o == 1) / len(filtered)
    
    def _calculate_rsi(self, numeric_outcomes: List[int], window: int) -> float:
        """
        Calculate Relative Strength Index (RSI) for player outcomes.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            window: RSI window size
            
        Returns:
            float: RSI value (0-1)
        """
        if len(numeric_outcomes) < window + 1:
            return 0.5
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < window + 1:
            return 0.5
            
        # Calculate gains and losses
        gains = []
        losses = []
        
        for i in range(1, len(filtered)):
            change = filtered[i] - filtered[i-1]
            if change > 0:  # Player after Banker = gain
                gains.append(1)
                losses.append(0)
            elif change < 0:  # Banker after Player = loss
                gains.append(0)
                losses.append(1)
            else:  # No change
                gains.append(0)
                losses.append(0)
        
        # Get the most recent window
        recent_gains = gains[-window:]
        recent_losses = losses[-window:]
        
        # Calculate average gain and loss
        avg_gain = sum(recent_gains) / window
        avg_loss = sum(recent_losses) / window
        
        # Calculate RSI
        if avg_loss == 0:
            return 1.0
            
        rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
        rsi = 1.0 - (1.0 / (1.0 + rs))
        
        return rsi
    
    def _calculate_macd(self, numeric_outcomes: List[int], 
                       fast_window: int, slow_window: int, signal_window: int) -> Tuple[float, float]:
        """
        Calculate Moving Average Convergence Divergence (MACD) for player outcomes.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            fast_window: Fast EMA window
            slow_window: Slow EMA window
            signal_window: Signal EMA window
            
        Returns:
            tuple: (MACD line, MACD histogram)
        """
        if len(numeric_outcomes) < slow_window + signal_window:
            return 0.0, 0.0
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < slow_window + signal_window:
            return 0.0, 0.0
            
        # Calculate EMAs
        def ema(data, window):
            alpha = 2.0 / (window + 1)
            ema_values = [data[0]]
            for i in range(1, len(data)):
                ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[i-1])
            return ema_values
        
        # Calculate fast and slow EMAs
        fast_ema = ema(filtered, fast_window)[-1]
        slow_ema = ema(filtered, slow_window)[-1]
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        macd_values = []
        for i in range(len(filtered) - slow_window + 1):
            fast_ema_i = ema(filtered[:i+slow_window], fast_window)[-1]
            slow_ema_i = ema(filtered[:i+slow_window], slow_window)[-1]
            macd_values.append(fast_ema_i - slow_ema_i)
        
        signal_line = ema(macd_values, signal_window)[-1]
        
        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line
        
        return macd_line, macd_histogram
    
    def _calculate_stochastic(self, numeric_outcomes: List[int], k_window: int, d_window: int) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator for player outcomes.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            k_window: %K window
            d_window: %D window
            
        Returns:
            tuple: (%K, %D)
        """
        if len(numeric_outcomes) < k_window + d_window:
            return 0.5, 0.5
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < k_window + d_window:
            return 0.5, 0.5
            
        # Calculate %K values
        k_values = []
        
        for i in range(len(filtered) - k_window + 1):
            window = filtered[i:i+k_window]
            highest = 1  # Max is always 1 (Player)
            lowest = 0   # Min is always 0 (Banker)
            current = window[-1]
            
            # Calculate %K
            k = (current - lowest) / (highest - lowest) if highest > lowest else 0.5
            k_values.append(k)
        
        # Calculate %D (simple moving average of %K)
        d_values = []
        
        for i in range(len(k_values) - d_window + 1):
            d = sum(k_values[i:i+d_window]) / d_window
            d_values.append(d)
        
        # Get current values
        k = k_values[-1]
        d = d_values[-1]
        
        return k, d
    
    def _detect_divergence(self, numeric_outcomes: List[int]) -> Tuple[bool, int]:
        """
        Detect divergence between price and momentum indicators.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (divergence_detected, divergence_direction)
        """
        if len(numeric_outcomes) < self.long_window * 2:
            return False, 0
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < self.long_window * 2:
            return False, 0
            
        # Calculate player frequencies for two consecutive periods
        prev_period = filtered[-self.long_window*2:-self.long_window]
        curr_period = filtered[-self.long_window:]
        
        prev_freq = sum(1 for o in prev_period if o == 1) / len(prev_period)
        curr_freq = sum(1 for o in curr_period if o == 1) / len(curr_period)
        
        # Calculate momentum indicators for two consecutive periods
        prev_rsi = self._calculate_rsi(filtered[:-self.long_window], self.medium_window)
        curr_rsi = self._calculate_rsi(filtered, self.medium_window)
        
        # Detect divergence
        price_trend = 1 if curr_freq > prev_freq else -1
        momentum_trend = 1 if curr_rsi > prev_rsi else -1
        
        # Bearish divergence: price up, momentum down
        if price_trend == 1 and momentum_trend == -1:
            return True, -1
            
        # Bullish divergence: price down, momentum up
        if price_trend == -1 and momentum_trend == 1:
            return True, 1
            
        return False, 0
    
    def _calculate_trend(self, numeric_outcomes: List[int]) -> Tuple[int, float]:
        """
        Calculate current trend and its strength.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (trend_direction, trend_strength)
        """
        if len(numeric_outcomes) < self.long_window:
            return 0, 0.0
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < self.long_window:
            return 0, 0.0
            
        # Calculate player frequencies for different windows
        short_freq = self._calculate_player_frequency(filtered, self.short_window)
        medium_freq = self._calculate_player_frequency(filtered, self.medium_window)
        long_freq = self._calculate_player_frequency(filtered, self.long_window)
        
        # Calculate trend direction
        short_trend = 1 if short_freq > 0.5 else -1
        medium_trend = 1 if medium_freq > 0.5 else -1
        long_trend = 1 if long_freq > 0.5 else -1
        
        # Calculate trend strength based on alignment of different timeframes
        if short_trend == medium_trend == long_trend:
            # Strong trend - all timeframes aligned
            trend_direction = short_trend
            trend_strength = abs(short_freq - 0.5) * 2  # Scale to 0-1
        elif medium_trend == long_trend:
            # Moderate trend - medium and long aligned
            trend_direction = medium_trend
            trend_strength = abs(medium_freq - 0.5) * 1.5  # Scale to 0-1
        elif short_trend == medium_trend:
            # Weak trend - short and medium aligned
            trend_direction = short_trend
            trend_strength = abs(short_freq - 0.5)  # Scale to 0-1
        else:
            # No clear trend
            trend_direction = 0
            trend_strength = 0.0
        
        return trend_direction, min(1.0, trend_strength)
    
    def _detect_overbought_oversold(self, numeric_outcomes: List[int]) -> Tuple[bool, int]:
        """
        Detect overbought or oversold conditions.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            tuple: (condition_detected, condition_direction)
        """
        if len(numeric_outcomes) < self.medium_window:
            return False, 0
            
        # Calculate RSI
        rsi = self._calculate_rsi(numeric_outcomes, self.medium_window)
        
        # Calculate Stochastic
        k, d = self._calculate_stochastic(numeric_outcomes, self.short_window, self.short_window)
        
        # Detect overbought (favor Banker)
        if rsi > self.overbought_threshold and k > self.overbought_threshold:
            return True, -1
            
        # Detect oversold (favor Player)
        if rsi < self.oversold_threshold and k < self.oversold_threshold:
            return True, 1
            
        return False, 0
    
    def _calculate_momentum_indicators(self, numeric_outcomes: List[int]):
        """
        Calculate all momentum indicators.
        
        Args:
            numeric_outcomes: List of numeric outcomes
        """
        # Calculate player frequencies
        self.momentum_indicators['short_freq'] = self._calculate_player_frequency(numeric_outcomes, self.short_window)
        self.momentum_indicators['medium_freq'] = self._calculate_player_frequency(numeric_outcomes, self.medium_window)
        self.momentum_indicators['long_freq'] = self._calculate_player_frequency(numeric_outcomes, self.long_window)
        
        # Calculate RSI
        if self.use_rsi:
            self.momentum_indicators['rsi'] = self._calculate_rsi(numeric_outcomes, self.medium_window)
        
        # Calculate MACD
        if self.use_macd:
            macd_line, macd_hist = self._calculate_macd(numeric_outcomes, self.short_window, 
                                                      self.medium_window, self.short_window)
            self.momentum_indicators['macd_line'] = macd_line
            self.momentum_indicators['macd_hist'] = macd_hist
        
        # Calculate Stochastic
        if self.use_stochastic:
            k, d = self._calculate_stochastic(numeric_outcomes, self.short_window, self.short_window)
            self.momentum_indicators['stoch_k'] = k
            self.momentum_indicators['stoch_d'] = d
        
        # Detect trend
        trend_direction, trend_strength = self._calculate_trend(numeric_outcomes)
        self.current_trend = trend_direction
        self.trend_strength = trend_strength
        
        # Detect divergence
        divergence_detected, divergence_direction = self._detect_divergence(numeric_outcomes)
        self.momentum_indicators['divergence_detected'] = divergence_detected
        self.momentum_indicators['divergence_direction'] = divergence_direction
        
        # Detect overbought/oversold
        condition_detected, condition_direction = self._detect_overbought_oversold(numeric_outcomes)
        self.momentum_indicators['condition_detected'] = condition_detected
        self.momentum_indicators['condition_direction'] = condition_direction
    
    def _combine_signals(self) -> Dict[str, float]:
        """
        Combine all momentum signals into a final prediction.
        
        Returns:
            dict: Prediction probabilities
        """
        p_prob = 0.5
        b_prob = 0.5
        
        # Start with neutral probabilities
        signals = []
        
        # Trend signal
        if abs(self.current_trend) > 0 and self.trend_strength > self.trend_threshold:
            trend_signal = self.current_trend * self.trend_strength
            signals.append(('trend', trend_signal))
        
        # RSI signal
        if self.use_rsi and 'rsi' in self.momentum_indicators:
            rsi = self.momentum_indicators['rsi']
            rsi_signal = (rsi - 0.5) * 2  # Scale to [-1, 1]
            signals.append(('rsi', rsi_signal * self.rsi_weight))
        
        # MACD signal
        if self.use_macd and 'macd_hist' in self.momentum_indicators:
            macd_hist = self.momentum_indicators['macd_hist']
            macd_signal = np.clip(macd_hist * 5, -1, 1)  # Scale and clip to [-1, 1]
            signals.append(('macd', macd_signal * self.macd_weight))
        
        # Stochastic signal
        if self.use_stochastic and 'stoch_k' in self.momentum_indicators and 'stoch_d' in self.momentum_indicators:
            k = self.momentum_indicators['stoch_k']
            d = self.momentum_indicators['stoch_d']
            stoch_signal = ((k + d) / 2 - 0.5) * 2  # Scale to [-1, 1]
            signals.append(('stoch', stoch_signal * self.stochastic_weight))
        
        # Divergence signal (strong signal)
        if self.momentum_indicators.get('divergence_detected', False):
            div_direction = self.momentum_indicators['divergence_direction']
            signals.append(('divergence', div_direction * 0.5))  # Strong but not overwhelming
        
        # Overbought/oversold signal (strong signal)
        if self.momentum_indicators.get('condition_detected', False):
            cond_direction = self.momentum_indicators['condition_direction']
            signals.append(('condition', cond_direction * 0.6))  # Very strong signal
        
        # Combine all signals
        if signals:
            combined_signal = sum(weight for _, weight in signals) / len(signals)
            
            # Convert to probabilities
            signal_strength = abs(combined_signal)
            if combined_signal > 0:  # Favor Player
                p_prob = 0.5 + signal_strength * 0.5
                b_prob = 1.0 - p_prob
            else:  # Favor Banker
                b_prob = 0.5 + signal_strength * 0.5
                p_prob = 1.0 - b_prob
        
        # Apply banker bias
        b_prob += b_prob * self.banker_bias
        
        # Normalize
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        return {'P': p_prob, 'B': b_prob}
    
    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using multi-scale momentum analysis.
        
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
        
        # Not enough data for momentum analysis
        if len(numeric) < self.min_samples:
            return 'B'  # Default to Banker
        
        # Calculate momentum indicators
        self._calculate_momentum_indicators(numeric)
        
        # Combine signals
        probs = self._combine_signals()
        
        # Make decision
        return 'P' if probs['P'] > probs['B'] else 'B'
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.
        
        Returns:
            dict: Strategy statistics
        """
        stats = {
            "strategy": "Multi-Scale Momentum",
            "trend": {
                "direction": "Player" if self.current_trend == 1 else "Banker" if self.current_trend == -1 else "None",
                "strength": f"{self.trend_strength:.2f}"
            },
            "frequencies": {
                "short": f"{self.momentum_indicators.get('short_freq', 0):.2f}",
                "medium": f"{self.momentum_indicators.get('medium_freq', 0):.2f}",
                "long": f"{self.momentum_indicators.get('long_freq', 0):.2f}"
            }
        }
        
        # Add RSI
        if self.use_rsi and 'rsi' in self.momentum_indicators:
            stats["rsi"] = f"{self.momentum_indicators['rsi']:.2f}"
        
        # Add MACD
        if self.use_macd and 'macd_line' in self.momentum_indicators:
            stats["macd"] = {
                "line": f"{self.momentum_indicators['macd_line']:.4f}",
                "histogram": f"{self.momentum_indicators['macd_hist']:.4f}"
            }
        
        # Add Stochastic
        if self.use_stochastic and 'stoch_k' in self.momentum_indicators:
            stats["stochastic"] = {
                "k": f"{self.momentum_indicators['stoch_k']:.2f}",
                "d": f"{self.momentum_indicators['stoch_d']:.2f}"
            }
        
        # Add divergence
        if self.momentum_indicators.get('divergence_detected', False):
            stats["divergence"] = {
                "detected": True,
                "direction": "Bullish (Player)" if self.momentum_indicators['divergence_direction'] == 1 else "Bearish (Banker)"
            }
        
        # Add overbought/oversold
        if self.momentum_indicators.get('condition_detected', False):
            condition = "Oversold (Player)" if self.momentum_indicators['condition_direction'] == 1 else "Overbought (Banker)"
            stats["condition"] = condition
        
        return stats
