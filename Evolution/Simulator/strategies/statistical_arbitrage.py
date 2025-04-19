"""
Statistical Arbitrage Strategy

This strategy attempts to identify and exploit statistical inefficiencies
in the baccarat game by analyzing deviations from expected probabilities.
"""

import numpy as np
from collections import defaultdict, deque
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class StatisticalArbitrageStrategy:
    """
    A strategy that uses statistical arbitrage principles to identify and exploit
    temporary deviations from expected probabilities in baccarat outcomes.
    
    Features:
    - Z-score calculation for detecting statistical anomalies
    - Mean reversion and momentum detection
    - Dynamic threshold adjustment
    - Multi-window analysis
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Statistical Arbitrage strategy.
        
        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}
        
        # Core parameters
        self.short_window = params.get('short_window', 5)
        self.medium_window = params.get('medium_window', 8)
        self.long_window = params.get('long_window', 12)
        self.z_score_threshold = params.get('z_score_threshold', 1.5)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.min_samples = params.get('min_samples', 5)
        
        # Advanced parameters
        self.use_mean_reversion = params.get('use_mean_reversion', True)
        self.use_momentum = params.get('use_momentum', True)
        self.mean_reversion_weight = params.get('mean_reversion_weight', 0.6)
        self.momentum_weight = params.get('momentum_weight', 0.4)
        self.dynamic_threshold = params.get('dynamic_threshold', True)
        self.threshold_adjustment_rate = params.get('threshold_adjustment_rate', 0.1)
        self.use_kelly_criterion = params.get('use_kelly_criterion', False)
        
        # Expected probabilities (theoretical)
        self.expected_p_prob = 0.4462  # Player win probability
        self.expected_b_prob = 0.4585  # Banker win probability
        self.expected_t_prob = 0.0953  # Tie probability
        
        # State tracking
        self.numeric_history = []  # 1 for Player, 0 for Banker, -1 for Tie
        self.recent_performance = deque(maxlen=50)
        self.z_scores = {'short': 0, 'medium': 0, 'long': 0}
        self.current_threshold = self.z_score_threshold
        self.current_edge = 0.0
        
        # Pattern tracking
        self.pattern_stats = defaultdict(lambda: {'count': 0, 'p_next': 0, 'b_next': 0})
        
    def _outcomes_to_numeric(self, outcomes: List[str]) -> List[int]:
        """
        Convert outcome strings to numeric values.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            list: Numeric outcomes (1 for P, 0 for B, -1 for T)
        """
        return [1 if o == 'P' else (0 if o == 'B' else -1) for o in outcomes]
    
    def _calculate_z_score(self, numeric_outcomes: List[int], window_size: int) -> float:
        """
        Calculate z-score for player frequency in the given window.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            window_size: Size of the window to analyze
            
        Returns:
            float: Z-score value
        """
        if len(numeric_outcomes) < window_size:
            return 0.0
            
        # Get the most recent outcomes in the window
        recent = numeric_outcomes[-window_size:]
        
        # Filter out ties
        filtered = [o for o in recent if o != -1]
        
        if len(filtered) < self.min_samples:
            return 0.0
            
        # Calculate player frequency
        p_freq = sum(1 for o in filtered if o == 1) / len(filtered)
        
        # Expected probability and standard deviation
        p = self.expected_p_prob / (self.expected_p_prob + self.expected_b_prob)  # Normalized for non-tie outcomes
        std_dev = np.sqrt(p * (1 - p) / len(filtered))
        
        # Calculate z-score
        if std_dev == 0:
            return 0.0
            
        return (p_freq - p) / std_dev
    
    def _detect_patterns(self, numeric_outcomes: List[int], pattern_length: int = 3) -> Dict[str, float]:
        """
        Detect patterns in the outcome history and their predictive power.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            pattern_length: Length of patterns to analyze
            
        Returns:
            dict: Pattern predictions with confidence scores
        """
        if len(numeric_outcomes) < pattern_length + 1:
            return {'P': 0.5, 'B': 0.5}
            
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < pattern_length + 1:
            return {'P': 0.5, 'B': 0.5}
            
        # Update pattern statistics
        for i in range(len(filtered) - pattern_length):
            pattern = tuple(filtered[i:i+pattern_length])
            next_outcome = filtered[i+pattern_length]
            
            self.pattern_stats[pattern]['count'] += 1
            if next_outcome == 1:  # Player
                self.pattern_stats[pattern]['p_next'] += 1
            else:  # Banker
                self.pattern_stats[pattern]['b_next'] += 1
        
        # Get current pattern
        current_pattern = tuple(filtered[-pattern_length:])
        
        # Check if we have statistics for this pattern
        if current_pattern in self.pattern_stats and self.pattern_stats[current_pattern]['count'] >= self.min_samples:
            stats = self.pattern_stats[current_pattern]
            p_prob = stats['p_next'] / stats['count']
            b_prob = stats['b_next'] / stats['count']
            
            # Apply banker bias
            b_prob += b_prob * self.banker_bias
            
            # Normalize
            total = p_prob + b_prob
            p_prob /= total
            b_prob /= total
            
            return {'P': p_prob, 'B': b_prob}
        else:
            return {'P': 0.5, 'B': 0.5}
    
    def _calculate_edge(self, z_scores: Dict[str, float], pattern_probs: Dict[str, float]) -> Tuple[str, float]:
        """
        Calculate statistical edge based on z-scores and pattern probabilities.
        
        Args:
            z_scores: Z-scores for different windows
            pattern_probs: Pattern-based probabilities
            
        Returns:
            tuple: (bet, edge)
        """
        # Initialize
        p_edge = 0.0
        b_edge = 0.0
        
        # Mean reversion component
        if self.use_mean_reversion:
            # Positive z-score means player is overrepresented, so bet on banker
            mean_reversion_signal = -np.mean([z_scores['short'], z_scores['medium'], z_scores['long']])
            
            # Scale to [-1, 1] range
            mean_reversion_signal = np.clip(mean_reversion_signal / 3.0, -1.0, 1.0)
            
            # Positive signal favors player, negative favors banker
            if mean_reversion_signal > 0:
                p_edge += mean_reversion_signal * self.mean_reversion_weight
            else:
                b_edge += -mean_reversion_signal * self.mean_reversion_weight
        
        # Momentum component
        if self.use_momentum:
            # Short-term momentum
            momentum_signal = z_scores['short'] - z_scores['medium']
            
            # Scale to [-1, 1] range
            momentum_signal = np.clip(momentum_signal / 2.0, -1.0, 1.0)
            
            # Positive signal favors player, negative favors banker
            if momentum_signal > 0:
                p_edge += momentum_signal * self.momentum_weight
            else:
                b_edge += -momentum_signal * self.momentum_weight
        
        # Pattern component
        pattern_signal = pattern_probs['P'] - pattern_probs['B']
        
        # Add pattern signal
        if pattern_signal > 0:
            p_edge += pattern_signal * 0.5  # Weight pattern less than statistical signals
        else:
            b_edge += -pattern_signal * 0.5
        
        # Apply banker bias
        b_edge += self.banker_bias
        
        # Determine bet and edge
        if p_edge > b_edge:
            return 'P', p_edge - b_edge
        else:
            return 'B', b_edge - p_edge
    
    def _adjust_threshold(self, success: bool):
        """
        Dynamically adjust z-score threshold based on recent performance.
        
        Args:
            success: Whether the last bet was successful
        """
        if not self.dynamic_threshold:
            return
            
        # Adjust threshold based on success/failure
        if success:
            # If successful, slightly decrease threshold to be more aggressive
            self.current_threshold = max(0.5, self.current_threshold - self.threshold_adjustment_rate)
        else:
            # If unsuccessful, increase threshold to be more conservative
            self.current_threshold = min(3.0, self.current_threshold + self.threshold_adjustment_rate)
    
    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using statistical arbitrage principles.
        
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
        
        # Not enough data for statistical analysis
        if len(numeric) < self.min_samples:
            return 'B'  # Default to Banker
        
        # Calculate z-scores for different windows
        self.z_scores = {
            'short': self._calculate_z_score(numeric, self.short_window),
            'medium': self._calculate_z_score(numeric, self.medium_window),
            'long': self._calculate_z_score(numeric, self.long_window)
        }
        
        # Detect patterns
        pattern_probs = self._detect_patterns(numeric)
        
        # Calculate edge
        bet, edge = self._calculate_edge(self.z_scores, pattern_probs)
        self.current_edge = edge
        
        # Only bet if edge exceeds threshold
        if edge < self.current_threshold / 10.0:  # Scale threshold to reasonable edge values
            # Not enough edge, default to banker
            return 'B'
            
        return bet
    
    def update_result(self, bet: str, outcome: str, win: bool):
        """
        Update strategy with the result of the last bet.
        
        Args:
            bet: The bet that was placed ('P' or 'B')
            outcome: The actual outcome ('P', 'B', or 'T')
            win: Whether the bet won
        """
        # Track performance for threshold adjustment
        if outcome != 'T':  # Ignore ties
            self.recent_performance.append(win)
            self._adjust_threshold(win)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.
        
        Returns:
            dict: Strategy statistics
        """
        # Calculate recent performance
        win_rate = sum(self.recent_performance) / len(self.recent_performance) if self.recent_performance else 0
        
        # Find strongest patterns
        strong_patterns = {}
        for pattern, stats in self.pattern_stats.items():
            if stats['count'] >= self.min_samples:
                p_prob = stats['p_next'] / stats['count']
                b_prob = stats['b_next'] / stats['count']
                max_prob = max(p_prob, b_prob)
                if max_prob > 0.65:  # Only include strong patterns
                    pattern_str = ''.join(['P' if p == 1 else 'B' for p in pattern])
                    strong_patterns[pattern_str] = {
                        'count': stats['count'],
                        'P_next': f"{p_prob:.2f}",
                        'B_next': f"{b_prob:.2f}"
                    }
        
        return {
            "strategy": "Statistical Arbitrage",
            "z_scores": {k: f"{v:.2f}" for k, v in self.z_scores.items()},
            "current_threshold": f"{self.current_threshold:.2f}",
            "current_edge": f"{self.current_edge:.4f}",
            "win_rate": f"{win_rate:.2f}",
            "strong_patterns": strong_patterns,
            "total_patterns": len(self.pattern_stats)
        }
