"""
Advanced Chaos Theory strategy implementation with additional features.

This extends the original ChaosTheoryStrategy with more sophisticated
chaos theory techniques that can be toggled on/off.
"""

import logging
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy
from .chaos_theory import ChaosTheoryStrategy

logger = logging.getLogger(__name__)

class ChaosTheoryAdvancedStrategy(ChaosTheoryStrategy):
    """
    Advanced Chaos Theory strategy that extends the original implementation
    with additional techniques from chaos theory and nonlinear dynamics.
    
    New features include:
    - Recurrence Quantification Analysis (RQA)
    - Fractal dimension calculation
    - Entropy-based prediction
    - Adaptive embedding parameters
    - Ensemble prediction methods
    
    All advanced features can be toggled on/off.
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Advanced Chaos Theory strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)

        # Advanced feature toggles
        self.use_rqa = params.get('use_rqa', False)
        self.use_fractal = params.get('use_fractal', False)
        self.use_entropy = params.get('use_entropy', False)
        self.use_adaptive_embedding = params.get('use_adaptive_embedding', False)
        self.use_ensemble = params.get('use_ensemble', False)
        
        # Advanced parameters
        self.rqa_threshold = params.get('rqa_threshold', 0.1)
        self.rqa_min_diagonal = params.get('rqa_min_diagonal', 2)
        self.entropy_window = params.get('entropy_window', 5)
        self.fractal_scales = params.get('fractal_scales', [1, 2, 3, 4])
        self.ensemble_weights = params.get('ensemble_weights', [0.5, 0.2, 0.15, 0.1, 0.05])
        
        # Adaptive embedding parameters
        self.min_embedding_dimension = params.get('min_embedding_dimension', 2)
        self.max_embedding_dimension = params.get('max_embedding_dimension', 6)
        self.min_time_delay = params.get('min_time_delay', 1)
        self.max_time_delay = params.get('max_time_delay', 10)
        
        # Additional tracking variables
        self.recurrence_matrix = None
        self.fractal_dimension = 0.0
        self.sample_entropy = 0.0
        self.prediction_methods = []
        
        logger.info(f"Initialized Advanced Chaos Theory strategy with features: "
                   f"RQA={self.use_rqa}, Fractal={self.use_fractal}, "
                   f"Entropy={self.use_entropy}, Adaptive={self.use_adaptive_embedding}, "
                   f"Ensemble={self.use_ensemble}")

    def _create_recurrence_plot(self, phase_space, threshold=None):
        """
        Create a recurrence plot from phase space data.
        
        Args:
            phase_space: Reconstructed phase space
            threshold: Distance threshold for recurrence (if None, use self.rqa_threshold)
            
        Returns:
            numpy.ndarray: Recurrence matrix (binary)
        """
        if len(phase_space) < 2:
            return np.array([])
            
        threshold = threshold or self.rqa_threshold
        
        # Calculate distance matrix
        dist_matrix = squareform(pdist(phase_space, 'euclidean'))
        
        # Create recurrence matrix using threshold
        recurrence_matrix = dist_matrix < threshold
        
        return recurrence_matrix
        
    def _calculate_rqa_metrics(self, recurrence_matrix):
        """
        Calculate Recurrence Quantification Analysis metrics.
        
        Args:
            recurrence_matrix: Binary recurrence matrix
            
        Returns:
            dict: RQA metrics including recurrence rate, determinism, and average diagonal length
        """
        if len(recurrence_matrix) == 0:
            return {'recurrence_rate': 0, 'determinism': 0, 'avg_diagonal_length': 0}
            
        # Recurrence Rate (RR)
        recurrence_rate = np.mean(recurrence_matrix)
        
        # Find diagonal lines (determinism)
        diagonals = []
        min_diagonal = self.rqa_min_diagonal
        n = len(recurrence_matrix)
        
        for i in range(-(n-min_diagonal), n-min_diagonal+1):
            diagonal = np.diag(recurrence_matrix, k=i)
            
            # Count consecutive True values
            if len(diagonal) >= min_diagonal:
                count = 0
                for val in diagonal:
                    if val:
                        count += 1
                    else:
                        if count >= min_diagonal:
                            diagonals.append(count)
                        count = 0
                        
                # Check the last sequence
                if count >= min_diagonal:
                    diagonals.append(count)
        
        # Calculate determinism and average diagonal length
        if diagonals:
            determinism = sum(diagonals) / np.sum(recurrence_matrix)
            avg_diagonal_length = np.mean(diagonals)
        else:
            determinism = 0
            avg_diagonal_length = 0
            
        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'avg_diagonal_length': avg_diagonal_length
        }
        
    def _calculate_fractal_dimension(self, time_series):
        """
        Estimate the correlation dimension (a type of fractal dimension).
        
        Args:
            time_series: Numeric time series data
            
        Returns:
            float: Estimated correlation dimension
        """
        if len(time_series) < 20:  # Need sufficient data
            return 0.0
            
        # Use multiple embedding dimensions to estimate correlation dimension
        dimensions = []
        
        for dim in range(self.min_embedding_dimension, self.max_embedding_dimension + 1):
            # Create phase space with current dimension
            phase_space = self._reconstruct_phase_space_with_dimension(time_series, dim, self.time_delay)
            
            if len(phase_space) < 10:
                continue
                
            # Calculate pairwise distances
            distances = pdist(phase_space, 'euclidean')
            
            if len(distances) == 0:
                continue
                
            # Calculate correlation sum for different scales
            correlation_sums = []
            
            for scale in self.fractal_scales:
                epsilon = np.std(distances) * scale / 10.0
                correlation_sum = np.sum(distances < epsilon) / (len(distances) * 2)
                correlation_sums.append(correlation_sum)
                
            # Estimate dimension from log-log relationship
            if min(correlation_sums) > 0:
                log_scales = np.log([np.std(distances) * s / 10.0 for s in self.fractal_scales])
                log_sums = np.log(correlation_sums)
                
                try:
                    slope, _ = np.polyfit(log_scales, log_sums, 1)
                    dimensions.append(slope)
                except:
                    pass
        
        # Return average dimension if available
        return np.mean(dimensions) if dimensions else 0.0
        
    def _reconstruct_phase_space_with_dimension(self, time_series, dimension, time_delay):
        """
        Reconstruct phase space with specific dimension and time delay.
        
        Args:
            time_series: Numeric time series data
            dimension: Embedding dimension
            time_delay: Time delay
            
        Returns:
            numpy.ndarray: Reconstructed phase space
        """
        if len(time_series) < dimension * time_delay:
            return np.array([])
            
        n = len(time_series) - (dimension - 1) * time_delay
        phase_space = np.zeros((n, dimension))
        
        for i in range(n):
            for j in range(dimension):
                phase_space[i, j] = time_series[i + j * time_delay]
                
        return phase_space
        
    def _calculate_sample_entropy(self, time_series, window=None):
        """
        Calculate sample entropy of the time series.
        
        Args:
            time_series: Numeric time series data
            window: Window size for entropy calculation
            
        Returns:
            float: Sample entropy value
        """
        window = window or self.entropy_window
        
        if len(time_series) < window * 2:
            return 0.0
            
        # Calculate probabilities of patterns
        pattern_counts = defaultdict(int)
        total_patterns = len(time_series) - window + 1
        
        for i in range(total_patterns):
            pattern = tuple(time_series[i:i+window])
            pattern_counts[pattern] += 1
            
        # Calculate probabilities
        probabilities = [count / total_patterns for count in pattern_counts.values()]
        
        # Calculate entropy
        return entropy(probabilities)
        
    def _find_optimal_embedding_parameters(self, time_series):
        """
        Find optimal embedding dimension and time delay using mutual information
        and false nearest neighbors methods.
        
        Args:
            time_series: Numeric time series data
            
        Returns:
            tuple: (optimal_dimension, optimal_delay)
        """
        if len(time_series) < 20:
            return self.embedding_dimension, self.time_delay
            
        # Simple heuristic for time delay: first minimum of autocorrelation
        optimal_delay = self.time_delay
        
        for delay in range(self.min_time_delay, min(self.max_time_delay, len(time_series) // 4)):
            if delay >= len(time_series) - 1:
                break
                
            # Calculate autocorrelation
            series1 = time_series[:-delay]
            series2 = time_series[delay:]
            
            # Ensure same length
            min_len = min(len(series1), len(series2))
            series1 = series1[:min_len]
            series2 = series2[:min_len]
            
            if len(series1) < 2:
                continue
                
            # Calculate correlation
            try:
                correlation = np.corrcoef(series1, series2)[0, 1]
                
                # First minimum or when correlation crosses zero
                if np.isnan(correlation):
                    continue
                    
                if delay > 1 and correlation < 0:
                    optimal_delay = delay
                    break
            except:
                pass
                
        # Simple heuristic for embedding dimension: false nearest neighbors
        optimal_dimension = self.embedding_dimension
        
        for dim in range(self.min_embedding_dimension, self.max_embedding_dimension + 1):
            # Create phase spaces with consecutive dimensions
            phase_space1 = self._reconstruct_phase_space_with_dimension(time_series, dim, optimal_delay)
            phase_space2 = self._reconstruct_phase_space_with_dimension(time_series, dim + 1, optimal_delay)
            
            if len(phase_space1) < 10 or len(phase_space2) < 10:
                continue
                
            # Truncate to same length
            min_len = min(len(phase_space1), len(phase_space2))
            phase_space1 = phase_space1[:min_len]
            
            # For each point, find nearest neighbor in dim and check if it's still a neighbor in dim+1
            false_neighbors_ratio = 0
            
            try:
                for i in range(min_len):
                    # Find nearest neighbor (excluding self)
                    distances = np.array([np.linalg.norm(phase_space1[i] - phase_space1[j]) 
                                         for j in range(min_len) if j != i])
                    nearest_idx = np.argmin(distances)
                    
                    # Adjust index if it's after i
                    if nearest_idx >= i:
                        nearest_idx += 1
                        
                    # Check distance in higher dimension
                    distance_dim = distances[nearest_idx]
                    distance_dim_plus_1 = np.linalg.norm(phase_space2[i] - phase_space2[nearest_idx])
                    
                    # If distance increases significantly, it's a false neighbor
                    if distance_dim > 0 and (distance_dim_plus_1 / distance_dim) > 2.0:
                        false_neighbors_ratio += 1 / min_len
                        
                # If false neighbors ratio is small enough, we found our dimension
                if false_neighbors_ratio < 0.1:
                    optimal_dimension = dim
                    break
            except:
                pass
                
        return optimal_dimension, optimal_delay
        
    def _ensemble_prediction(self, time_series, phase_space):
        """
        Combine multiple prediction methods for a more robust forecast.
        
        Args:
            time_series: Numeric time series data
            phase_space: Reconstructed phase space
            
        Returns:
            tuple: (p_prob, b_prob) - Prediction probabilities
        """
        predictions = []
        
        # Method 1: Original nearest neighbors prediction
        current_point = phase_space[-1] if len(phase_space) > 0 else None
        
        if current_point is not None:
            neighbors = self._find_nearest_neighbors(current_point, phase_space)
            p_prob1, b_prob1 = self._predict_from_neighbors(neighbors, time_series)
            predictions.append((p_prob1, b_prob1))
        else:
            predictions.append((0.5, 0.5))
            
        # Method 2: Frequency analysis with recent bias
        if len(time_series) > 0:
            recent = time_series[-min(20, len(time_series)):]
            p_count = sum(1 for x in recent if x == 1)
            b_count = len(recent) - p_count
            
            p_prob2 = p_count / len(recent) if recent else 0.5
            b_prob2 = b_count / len(recent) if recent else 0.5
            
            predictions.append((p_prob2, b_prob2))
        else:
            predictions.append((0.5, 0.5))
            
        # Method 3: Pattern matching
        if len(time_series) >= 5:
            last_pattern = time_series[-3:]
            matches = []
            
            for i in range(len(time_series) - 4):
                if time_series[i:i+3] == last_pattern:
                    matches.append(time_series[i+3])
                    
            if matches:
                p_prob3 = sum(1 for x in matches if x == 1) / len(matches)
                b_prob3 = 1 - p_prob3
                predictions.append((p_prob3, b_prob3))
            else:
                predictions.append((0.5, 0.5))
        else:
            predictions.append((0.5, 0.5))
            
        # Method 4: RQA-based prediction if enabled
        if self.use_rqa and self.recurrence_matrix is not None and len(self.recurrence_matrix) > 0:
            try:
                # Find similar states in recurrence matrix
                current_idx = len(self.recurrence_matrix) - 1
                similar_states = np.where(self.recurrence_matrix[current_idx])[0]
                
                if len(similar_states) > 0:
                    # Get next states after similar states
                    next_states = [time_series[i+1] for i in similar_states if i+1 < len(time_series)]
                    
                    if next_states:
                        p_prob4 = sum(1 for x in next_states if x == 1) / len(next_states)
                        b_prob4 = 1 - p_prob4
                        predictions.append((p_prob4, b_prob4))
                    else:
                        predictions.append((0.5, 0.5))
                else:
                    predictions.append((0.5, 0.5))
            except:
                predictions.append((0.5, 0.5))
        else:
            predictions.append((0.5, 0.5))
            
        # Method 5: Entropy-based prediction if enabled
        if self.use_entropy and self.sample_entropy is not None:
            # Low entropy suggests more predictability, high entropy suggests randomness
            confidence = max(0, 1 - self.sample_entropy)
            
            # Use the most recent outcome with confidence based on entropy
            if len(time_series) > 0:
                last_outcome = time_series[-1]
                
                if last_outcome == 1:  # Player
                    p_prob5 = 0.5 + confidence * 0.5
                    b_prob5 = 1 - p_prob5
                else:  # Banker
                    b_prob5 = 0.5 + confidence * 0.5
                    p_prob5 = 1 - b_prob5
                    
                predictions.append((p_prob5, b_prob5))
            else:
                predictions.append((0.5, 0.5))
        else:
            predictions.append((0.5, 0.5))
            
        # Combine predictions using weights
        weights = self.ensemble_weights[:len(predictions)]
        
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        # Weighted average
        p_prob = sum(w * p for w, (p, _) in zip(weights, predictions))
        b_prob = sum(w * b for w, (_, b) in zip(weights, predictions))
        
        return p_prob, b_prob
        
    def get_bet(self, outcomes):
        """
        Determine the next bet using advanced chaos theory analysis.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        # If all advanced features are disabled, use the original implementation
        if not any([self.use_rqa, self.use_fractal, self.use_entropy, 
                   self.use_adaptive_embedding, self.use_ensemble]):
            return super().get_bet(outcomes)
            
        # Validate outcomes
        self._validate_outcome_list(outcomes)

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
            
        # Adaptive embedding if enabled
        if self.use_adaptive_embedding and len(numeric) >= 20:
            optimal_dimension, optimal_delay = self._find_optimal_embedding_parameters(numeric)
            
            # Temporarily update parameters for this prediction
            original_dimension = self.embedding_dimension
            original_delay = self.time_delay
            
            self.embedding_dimension = optimal_dimension
            self.time_delay = optimal_delay

        # Reconstruct phase space
        phase_space = self._reconstruct_phase_space(numeric)

        if len(phase_space) == 0:
            # Restore original parameters if they were changed
            if self.use_adaptive_embedding and len(numeric) >= 20:
                self.embedding_dimension = original_dimension
                self.time_delay = original_delay
                
            return 'B'  # Default to Banker if phase space reconstruction fails

        # Calculate RQA metrics if enabled
        if self.use_rqa:
            self.recurrence_matrix = self._create_recurrence_plot(phase_space)
            rqa_metrics = self._calculate_rqa_metrics(self.recurrence_matrix)
        else:
            rqa_metrics = {'recurrence_rate': 0, 'determinism': 0, 'avg_diagonal_length': 0}
            
        # Calculate fractal dimension if enabled
        if self.use_fractal:
            self.fractal_dimension = self._calculate_fractal_dimension(numeric)
        else:
            self.fractal_dimension = 0.0
            
        # Calculate entropy if enabled
        if self.use_entropy:
            self.sample_entropy = self._calculate_sample_entropy(numeric)
        else:
            self.sample_entropy = 0.0

        # Get current point in phase space
        current_point = phase_space[-1] if len(phase_space) > 0 else None

        # Estimate Lyapunov exponent
        if current_point is not None:
            neighbors = self._find_nearest_neighbors(current_point, phase_space)
            lyapunov = self._calculate_lyapunov_exponent(numeric, neighbors)
        else:
            neighbors = []
            lyapunov = 0.0

        # Make prediction
        if self.use_ensemble:
            # Use ensemble prediction
            p_prob, b_prob = self._ensemble_prediction(numeric, phase_space)
        else:
            # Use original prediction method
            p_prob, b_prob = self._predict_from_neighbors(neighbors, numeric)

            # Adjust prediction based on Lyapunov exponent
            confidence_factor = max(0.5, 1.0 - abs(lyapunov))
            
            # Adjust probabilities
            p_prob = 0.5 + (p_prob - 0.5) * confidence_factor
            b_prob = 0.5 + (b_prob - 0.5) * confidence_factor
            
            # Adjust based on RQA if enabled
            if self.use_rqa:
                # Higher determinism suggests more predictable patterns
                rqa_confidence = min(1.0, rqa_metrics['determinism'] * 2)
                
                p_prob = 0.5 + (p_prob - 0.5) * (1 + rqa_confidence)
                b_prob = 0.5 + (b_prob - 0.5) * (1 + rqa_confidence)
                
            # Adjust based on fractal dimension if enabled
            if self.use_fractal:
                # Lower fractal dimension suggests more order
                fractal_factor = max(0.5, 1.0 - min(1.0, self.fractal_dimension / 2.0))
                
                p_prob = 0.5 + (p_prob - 0.5) * fractal_factor
                b_prob = 0.5 + (b_prob - 0.5) * fractal_factor
                
            # Adjust based on entropy if enabled
            if self.use_entropy:
                # Lower entropy suggests more predictability
                entropy_factor = max(0.5, 1.0 - min(1.0, self.sample_entropy))
                
                p_prob = 0.5 + (p_prob - 0.5) * entropy_factor
                b_prob = 0.5 + (b_prob - 0.5) * entropy_factor

        # Apply banker bias
        b_prob += self.banker_bias

        # Normalize probabilities
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        # Restore original parameters if they were changed
        if self.use_adaptive_embedding and len(numeric) >= 20:
            self.embedding_dimension = original_dimension
            self.time_delay = original_delay

        # Return the bet with higher probability
        return 'P' if p_prob > b_prob else 'B'

    def get_stats(self):
        """Get strategy statistics for debugging."""
        stats = super().get_stats()
        
        # Add advanced stats
        if self.use_rqa:
            stats["rqa_enabled"] = True
            if self.recurrence_matrix is not None:
                rqa_metrics = self._calculate_rqa_metrics(self.recurrence_matrix)
                stats.update({
                    "recurrence_rate": rqa_metrics['recurrence_rate'],
                    "determinism": rqa_metrics['determinism'],
                    "avg_diagonal_length": rqa_metrics['avg_diagonal_length']
                })
                
        if self.use_fractal:
            stats["fractal_enabled"] = True
            stats["fractal_dimension"] = self.fractal_dimension
            
        if self.use_entropy:
            stats["entropy_enabled"] = True
            stats["sample_entropy"] = self.sample_entropy
            
        if self.use_adaptive_embedding:
            stats["adaptive_embedding_enabled"] = True
            
        if self.use_ensemble:
            stats["ensemble_enabled"] = True
            
        return stats
