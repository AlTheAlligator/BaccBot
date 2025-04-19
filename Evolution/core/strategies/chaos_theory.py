"""
Chaos Theory strategy implementation.
"""

import logging
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class ChaosTheoryStrategy():
    """
    Chaos Theory strategy that looks for deterministic patterns in seemingly random sequences.

    This strategy applies concepts from chaos theory like phase space reconstruction
    to identify hidden patterns in the outcome sequence that might appear random
    but have deterministic components.
    """

    def __init__(self, params=None):
        """
        Initialize the Chaos Theory strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        params = params or {}

        # Strategy parameters
        self.embedding_dimension = params.get('embedding_dimension', 4)
        self.time_delay = params.get('time_delay', 8)
        self.prediction_horizon = params.get('prediction_horizon', 12)
        self.num_neighbors = params.get('num_neighbors', 13)
        self.min_samples = params.get('min_samples', 7)  # Minimum samples before using chaos theory
        self.banker_bias = params.get('banker_bias', 0.06666666666666667)

        # Initialize tracking variables
        self.numeric_history = []  # 1 for Player, 0 for Banker

        logger.info(f"Initialized Chaos Theory strategy with embedding_dimension={self.embedding_dimension}")

    def _outcomes_to_numeric(self, outcomes):
        """
        Convert outcome strings to numeric values for analysis.

        Args:
            outcomes: List of outcomes ('P', 'B')

        Returns:
            list: Numeric values (1 for P, 0 for B)
        """
        return [1 if o == 'P' else 0 for o in outcomes]

    def _reconstruct_phase_space(self, time_series):
        """
        Reconstruct phase space from time series using time delay embedding.

        Args:
            time_series: Numeric time series data

        Returns:
            numpy.ndarray: Reconstructed phase space
        """
        if len(time_series) < self.embedding_dimension * self.time_delay:
            return np.array([])

        n = len(time_series) - (self.embedding_dimension - 1) * self.time_delay
        phase_space = np.zeros((n, self.embedding_dimension))

        for i in range(n):
            for j in range(self.embedding_dimension):
                phase_space[i, j] = time_series[i + j * self.time_delay]

        return phase_space

    def _find_nearest_neighbors(self, current_point, phase_space, exclude_last=True):
        """
        Find nearest neighbors to the current point in phase space.

        Args:
            current_point: Current point in phase space
            phase_space: Reconstructed phase space
            exclude_last: Whether to exclude the last point (current point)

        Returns:
            list: Indices of nearest neighbors
        """
        if len(phase_space) <= 1:
            return []

        # Calculate distances from current point to all points in phase space
        if exclude_last:
            distances = cdist([current_point], phase_space[:-1])[0]
            # Get indices of nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.num_neighbors]
            # Convert to list for consistency
            neighbor_indices = neighbor_indices.tolist()
        else:
            distances = cdist([current_point], phase_space)[0]
            # Get indices of nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.num_neighbors+1]
            # Convert to list for filtering
            neighbor_indices = neighbor_indices.tolist()
            # Remove current point if it's in the neighbors
            neighbor_indices = [i for i in neighbor_indices if float(distances[i]) > 0][:self.num_neighbors]

        return neighbor_indices

    def _predict_from_neighbors(self, neighbor_indices, time_series):
        """
        Predict next outcome based on neighbors' future states.

        Args:
            neighbor_indices: Indices of nearest neighbors
            time_series: Original time series data

        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        if not neighbor_indices or len(time_series) <= max(neighbor_indices or [0]) + self.prediction_horizon:
            return 0.5, 0.5

        # Get future states of neighbors
        future_states = [time_series[i + self.prediction_horizon] for i in neighbor_indices
                         if i + self.prediction_horizon < len(time_series)]

        if not future_states:
            return 0.5, 0.5

        # Count occurrences of each outcome
        p_count = sum(1 for state in future_states if state == 1)
        b_count = len(future_states) - p_count

        # Calculate probabilities
        p_prob = p_count / len(future_states) if future_states else 0.5
        b_prob = b_count / len(future_states) if future_states else 0.5

        return p_prob, b_prob

    def _calculate_lyapunov_exponent(self, time_series, neighbors, epsilon=1e-6, max_steps=10):
        """
        Estimate the largest Lyapunov exponent from time series data.

        Args:
            time_series: Numeric time series data
            neighbors: Indices of nearest neighbors
            epsilon: Small value to avoid log(0)
            max_steps: Maximum steps for divergence calculation

        Returns:
            float: Estimated largest Lyapunov exponent
        """
        if not neighbors or len(time_series) <= max(neighbors or [0]) + max_steps:
            return 0.0

        # Get current point index (assuming it's the last point in phase space)
        current_idx = len(time_series) - self.embedding_dimension * self.time_delay

        # Calculate divergence over time
        divergences = []
        for step in range(1, min(max_steps + 1, len(time_series) - max(neighbors) - 1)):
            # Current point future value
            current_future = time_series[current_idx + step] if current_idx + step < len(time_series) else None

            # Neighbors' future values
            neighbor_futures = [time_series[i + step] for i in neighbors if i + step < len(time_series)]

            if current_future is not None and neighbor_futures:
                # Calculate average divergence
                divergence = np.mean([abs(current_future - future) for future in neighbor_futures])
                divergences.append(np.log(divergence + epsilon))

        if not divergences:
            return 0.0

        # Estimate Lyapunov exponent as the slope of divergences
        steps = np.arange(1, len(divergences) + 1)

        # Handle potential numerical instability in polyfit
        try:
            # Suppress the RankWarning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                slope, _ = np.polyfit(steps, divergences, 1)
        except Exception as e:
            logger.debug(f"Error in polyfit: {e}")
            # Fallback to a simple linear regression
            if len(divergences) >= 2:
                slope = (divergences[-1] - divergences[0]) / (steps[-1] - steps[0])
            else:
                slope = 0.0

        return slope

    def get_bet(self, outcomes):
        """
        Determine the next bet using chaos theory analysis.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        # Always start from game 7
        if len(outcomes) < 7:
            return 'B'  # Default to Banker for initial games

        # Filter out ties and convert to numeric
        filtered = [o for o in outcomes if o in ['P', 'B']]
        numeric = self._outcomes_to_numeric(filtered)
        self.numeric_history = numeric

        # Not enough data for chaos analysis - use simple frequency analysis
        if len(numeric) < self.min_samples:
            p_count = numeric.count(1)
            b_count = numeric.count(0)

            # Apply banker bias
            b_count += b_count * self.banker_bias

            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'

        # Reconstruct phase space
        phase_space = self._reconstruct_phase_space(numeric)

        if len(phase_space) == 0:
            return 'B'  # Default to Banker if phase space reconstruction fails

        # Get current point in phase space
        current_point = phase_space[-1]

        # Find nearest neighbors
        neighbors = self._find_nearest_neighbors(current_point, phase_space)

        # Estimate Lyapunov exponent
        lyapunov = self._calculate_lyapunov_exponent(numeric, neighbors)

        # Predict next outcome based on neighbors
        p_prob, b_prob = self._predict_from_neighbors(neighbors, numeric)

        # Adjust prediction based on Lyapunov exponent
        # If lyapunov is high (chaotic), reduce confidence
        confidence_factor = max(0.5, 1.0 - abs(lyapunov))

        # Adjust probabilities
        p_prob = 0.5 + (p_prob - 0.5) * confidence_factor
        b_prob = 0.5 + (b_prob - 0.5) * confidence_factor

        # Apply banker bias
        b_prob += self.banker_bias

        # Normalize probabilities
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total

        # Return the bet with higher probability
        return 'P' if p_prob > b_prob else 'B'

    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "embedding_dimension": self.embedding_dimension,
            "time_delay": self.time_delay,
            "prediction_horizon": self.prediction_horizon,
            "num_neighbors": self.num_neighbors,
            "history_length": len(self.numeric_history)
        }
