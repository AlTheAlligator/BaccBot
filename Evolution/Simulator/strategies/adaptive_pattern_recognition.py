"""
Adaptive Pattern Recognition Strategy

This strategy uses machine learning techniques to identify recurring patterns
in the game outcomes and adapts its betting strategy accordingly.
"""

import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class AdaptivePatternRecognitionStrategy:
    """
    A strategy that uses adaptive pattern recognition to identify recurring patterns
    in baccarat outcomes and make predictions based on those patterns.

    Features:
    - Dynamic pattern length adjustment
    - Weighted pattern matching
    - Confidence-based betting
    - Adaptive learning rate
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Adaptive Pattern Recognition strategy.

        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}

        # Core parameters
        self.min_pattern_length = params.get('min_pattern_length', 2)
        self.max_pattern_length = params.get('max_pattern_length', 6)
        self.min_samples = params.get('min_samples', 5)
        self.confidence_threshold = params.get('confidence_threshold', 0.6)
        self.banker_bias = params.get('banker_bias', 0.01)  # Small bias for banker

        # Advanced parameters
        self.learning_rate = params.get('learning_rate', 0.1)
        self.decay_factor = params.get('decay_factor', 0.95)  # For recency weighting
        self.use_adaptive_patterns = params.get('use_adaptive_patterns', True)
        self.pattern_adjustment_threshold = params.get('pattern_adjustment_threshold', 0.3)
        self.use_weighted_voting = params.get('use_weighted_voting', True)
        self.weight_recent = params.get('weight_recent', 2.0)

        # Pattern storage
        self.pattern_stats = {}  # Stores pattern statistics
        self.recent_performance = []  # Tracks recent performance for adaptation
        self.current_pattern_length = self.min_pattern_length

        # Initialize pattern dictionaries for different lengths
        self.reset_pattern_stats()

    def reset_pattern_stats(self):
        """Reset pattern statistics."""
        self.pattern_stats = {}
        for length in range(self.min_pattern_length, self.max_pattern_length + 1):
            self.pattern_stats[length] = defaultdict(lambda: {'P': 0, 'B': 0, 'total': 0})

        self.recent_performance = []
        self.current_pattern_length = self.min_pattern_length

    def _outcomes_to_pattern(self, outcomes: List[str], length: int) -> str:
        """
        Convert a list of outcomes to a pattern string of specified length.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            length: Length of the pattern

        Returns:
            str: Pattern string
        """
        # Safety check for empty or invalid outcomes
        if not outcomes:
            return ""

        # Filter out ties and invalid values
        filtered = [o for o in outcomes if o in ['P', 'B']]

        if len(filtered) < length:
            return ""

        # Get the most recent outcomes of the specified length
        try:
            recent = filtered[-length:]
            return ''.join(recent)
        except Exception as e:
            logger.error(f"Error in _outcomes_to_pattern: {e}, outcomes={outcomes}, filtered={filtered}, length={length}")
            return ""

    def _update_pattern_stats(self, outcomes: List[str], new_outcome: str):
        """
        Update pattern statistics with a new outcome.

        Args:
            outcomes: List of previous outcomes
            new_outcome: The new outcome to record
        """
        # Safety check for invalid new_outcome
        if new_outcome not in ['P', 'B', 'T']:
            logger.warning(f"Invalid new_outcome in _update_pattern_stats: {new_outcome}")
            return

        try:
            # Update stats for each pattern length
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                pattern = self._outcomes_to_pattern(outcomes, length)
                if pattern:
                    # Apply decay to existing counts (recency weighting)
                    stats = self.pattern_stats[length][pattern]
                    stats['P'] *= self.decay_factor
                    stats['B'] *= self.decay_factor
                    stats['total'] *= self.decay_factor

                    # Update with new outcome
                    if new_outcome in ['P', 'B']:
                        stats[new_outcome] += 1
                        stats['total'] += 1
        except Exception as e:
            logger.error(f"Error in _update_pattern_stats: {e}, outcomes={outcomes}, new_outcome={new_outcome}")

    def _get_pattern_prediction(self, pattern: str, length: int) -> Tuple[str, float]:
        """
        Get prediction based on a specific pattern.

        Args:
            pattern: The pattern string
            length: Length of the pattern

        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Safety check for invalid inputs
            if not pattern or length <= 0 or length not in self.pattern_stats:
                return 'B', 0.5  # Default to banker with neutral confidence

            stats = self.pattern_stats[length][pattern]

            if stats['total'] < self.min_samples:
                return 'B', 0.5  # Default to banker with neutral confidence

            p_count = stats['P']
            b_count = stats['B']
            total = stats['total']

            # Apply banker bias
            b_count += b_count * self.banker_bias

            # Calculate probabilities
            p_prob = p_count / total if total > 0 else 0.5
            b_prob = b_count / total if total > 0 else 0.5

            # Normalize
            total_prob = p_prob + b_prob
            if total_prob > 0:  # Avoid division by zero
                p_prob /= total_prob
                b_prob /= total_prob
            else:
                p_prob, b_prob = 0.5, 0.5

            # Determine prediction and confidence
            if p_prob > b_prob:
                return 'P', p_prob
            else:
                return 'B', b_prob

        except Exception as e:
            logger.error(f"Error in _get_pattern_prediction: {e}, pattern={pattern}, length={length}")
            return 'B', 0.5  # Default to banker with neutral confidence

    def _adapt_pattern_length(self, outcomes: List[str]):
        """
        Adaptively adjust the pattern length based on recent performance.

        Args:
            outcomes: List of outcomes
        """
        try:
            # Safety check for invalid inputs
            if not outcomes or not self.use_adaptive_patterns or len(self.recent_performance) < 10:
                return

            # Calculate success rate for recent predictions
            success_rate = sum(self.recent_performance[-10:]) / 10

            # Adjust pattern length based on performance
            if success_rate < 0.4:  # Poor performance
                # Try a different pattern length
                if self.current_pattern_length == self.min_pattern_length:
                    self.current_pattern_length = self.max_pattern_length
                else:
                    self.current_pattern_length = max(self.current_pattern_length - 1, self.min_pattern_length)
            elif success_rate > 0.6:  # Good performance
                # Stick with current pattern length
                pass
            else:  # Moderate performance
                # Slightly adjust pattern length
                if np.random.random() < 0.3:
                    if np.random.random() < 0.5:
                        self.current_pattern_length = min(self.current_pattern_length + 1, self.max_pattern_length)
                    else:
                        self.current_pattern_length = max(self.current_pattern_length - 1, self.min_pattern_length)

        except Exception as e:
            logger.error(f"Error in _adapt_pattern_length: {e}, outcomes={outcomes}")

    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using adaptive pattern recognition.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        try:
            # Safety check for invalid inputs
            if not outcomes:
                return 'B'  # Default to Banker for empty outcomes

            # Always start from game 7
            if len(outcomes) < 7:
                return 'B'  # Default to Banker for initial games

            # Filter out ties and invalid values
            filtered = [o for o in outcomes if o in ['P', 'B']]

            if len(filtered) < self.min_pattern_length:
                return 'B'  # Not enough data, default to Banker

            # If we have a previous outcome, update pattern stats
            if len(filtered) > self.min_pattern_length:
                self._update_pattern_stats(filtered[:-1], filtered[-1])

            # Adapt pattern length based on recent performance
            self._adapt_pattern_length(filtered)

            # Get predictions for different pattern lengths
            predictions = []

            if self.use_weighted_voting:
                # Use all pattern lengths with weighted voting
                for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                    pattern = self._outcomes_to_pattern(filtered, length)
                    if pattern:
                        prediction, confidence = self._get_pattern_prediction(pattern, length)

                        # Weight by confidence and pattern length
                        weight = confidence * (length / self.max_pattern_length)

                        # Give more weight to recent patterns
                        if length == self.current_pattern_length:
                            weight *= self.weight_recent

                        predictions.append((prediction, weight))
            else:
                # Use only the current pattern length
                pattern = self._outcomes_to_pattern(filtered, self.current_pattern_length)
                if pattern:
                    prediction, confidence = self._get_pattern_prediction(pattern, self.current_pattern_length)
                    predictions.append((prediction, confidence))

            # Combine predictions
            if not predictions:
                return 'B'  # Default to Banker if no predictions

            # Calculate weighted votes
            p_votes = sum(weight for pred, weight in predictions if pred == 'P')
            b_votes = sum(weight for pred, weight in predictions if pred == 'B')

            # Apply banker bias
            b_votes += b_votes * self.banker_bias

            # Make final decision
            if p_votes > b_votes:
                return 'P'
            else:
                return 'B'

        except Exception as e:
            logger.error(f"Error in get_bet: {e}, outcomes={outcomes}")
            return 'B'  # Default to Banker in case of error

    def update_result(self, bet: str, outcome: str, win: bool):
        """
        Update strategy with the result of the last bet.

        Args:
            bet: The bet that was placed ('P' or 'B')
            outcome: The actual outcome ('P', 'B', or 'T')
            win: Whether the bet won
        """
        try:
            # Safety check for invalid inputs
            if bet not in ['P', 'B'] or outcome not in ['P', 'B', 'T']:
                logger.warning(f"Invalid inputs in update_result: bet={bet}, outcome={outcome}")
                return

            # Track performance for adaptation
            if outcome != 'T':  # Ignore ties
                self.recent_performance.append(1 if win else 0)

                # Keep only the most recent 50 results
                if len(self.recent_performance) > 50:
                    self.recent_performance = self.recent_performance[-50:]

        except Exception as e:
            logger.error(f"Error in update_result: {e}, bet={bet}, outcome={outcome}, win={win}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.

        Returns:
            dict: Strategy statistics
        """
        try:
            # Calculate pattern hit rates
            pattern_hit_rates = {}
            for length, patterns in self.pattern_stats.items():
                for pattern, stats in patterns.items():
                    if stats['total'] >= self.min_samples:
                        p_prob = stats['P'] / stats['total'] if stats['total'] > 0 else 0
                        b_prob = stats['B'] / stats['total'] if stats['total'] > 0 else 0
                        max_prob = max(p_prob, b_prob)
                        if max_prob > 0.6:  # Only include significant patterns
                            pattern_hit_rates[f"{pattern} (len={length})"] = {
                                'P': f"{p_prob:.2f}",
                                'B': f"{b_prob:.2f}",
                                'samples': stats['total']
                            }

            # Calculate recent performance
            recent_win_rate = sum(self.recent_performance[-20:]) / 20 if len(self.recent_performance) >= 20 else 0

            return {
                "strategy": "Adaptive Pattern Recognition",
                "current_pattern_length": self.current_pattern_length,
                "recent_win_rate": f"{recent_win_rate:.2f}",
                "significant_patterns": pattern_hit_rates,
                "total_patterns_tracked": sum(len(patterns) for patterns in self.pattern_stats.values())
            }

        except Exception as e:
            logger.error(f"Error in get_stats: {e}")
            return {
                "strategy": "Adaptive Pattern Recognition",
                "error": f"Error generating stats: {e}"
            }
