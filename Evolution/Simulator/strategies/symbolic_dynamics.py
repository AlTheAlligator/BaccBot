"""
Symbolic Dynamics strategy implementation.
"""

import logging
import numpy as np
from collections import defaultdict, deque
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SymbolicDynamicsStrategy(BaseStrategy):
    """
    Symbolic Dynamics strategy that applies concepts from chaos theory.
    
    This strategy converts outcome sequences into symbolic representations,
    analyzes them for forbidden patterns, calculates topological entropy,
    and uses recurrence plots to detect hidden patterns in the data.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Symbolic Dynamics strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.symbol_length = params.get('symbol_length', 3)
        self.recurrence_threshold = params.get('recurrence_threshold', 0.7)
        self.entropy_threshold = params.get('entropy_threshold', 0.8)
        self.forbidden_pattern_weight = params.get('forbidden_pattern_weight', 0.6)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.memory_length = params.get('memory_length', 200)
        
        # Initialize symbol storage
        self.symbol_counts = defaultdict(int)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.forbidden_patterns = set()
        
        # Initialize memory
        self.outcome_memory = deque(maxlen=self.memory_length)
        self.symbol_memory = deque(maxlen=self.memory_length)
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
    
    def _outcomes_to_symbols(self, outcomes, length):
        """
        Convert outcome sequence to symbolic representation.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            length: Length of symbols to generate
            
        Returns:
            list: List of symbols
        """
        if len(outcomes) < length:
            return []
        
        # Convert to binary (0 for B, 1 for P)
        binary = [1 if o == 'P' else 0 for o in outcomes]
        
        # Generate symbols
        symbols = []
        for i in range(len(binary) - length + 1):
            # Convert binary sequence to symbol (string)
            symbol = ''.join(str(b) for b in binary[i:i+length])
            symbols.append(symbol)
        
        return symbols
    
    def _update_symbol_statistics(self, outcomes):
        """
        Update symbol statistics based on outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Generate symbols
        symbols = self._outcomes_to_symbols(outcomes, self.symbol_length)
        
        # Update symbol counts
        for symbol in symbols:
            self.symbol_counts[symbol] += 1
        
        # Update transition matrix
        for i in range(len(symbols) - 1):
            current = symbols[i]
            next_symbol = symbols[i + 1]
            self.transition_matrix[current][next_symbol] += 1
        
        # Update symbol memory
        if symbols:
            self.symbol_memory.append(symbols[-1])
    
    def _identify_forbidden_patterns(self):
        """
        Identify patterns that never or rarely occur.
        
        Returns:
            set: Set of forbidden patterns
        """
        forbidden = set()
        
        # Generate all possible symbols of length symbol_length
        all_possible = []
        for i in range(2 ** self.symbol_length):
            # Convert number to binary string of length symbol_length
            binary = format(i, f'0{self.symbol_length}b')
            all_possible.append(binary)
        
        # Check which patterns never or rarely occur
        total_occurrences = sum(self.symbol_counts.values())
        if total_occurrences > 0:
            for symbol in all_possible:
                frequency = self.symbol_counts.get(symbol, 0) / total_occurrences
                if frequency < 0.01:  # Less than 1% occurrence
                    forbidden.add(symbol)
        
        return forbidden
    
    def _calculate_topological_entropy(self):
        """
        Calculate topological entropy of the symbolic dynamics.
        
        Returns:
            float: Entropy value between 0 and 1
        """
        # Count total symbol occurrences
        total = sum(self.symbol_counts.values())
        
        if total == 0:
            return 1.0  # Maximum entropy when no data
        
        # Calculate probabilities
        probabilities = [count / total for count in self.symbol_counts.values()]
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(2 ** self.symbol_length, len(self.symbol_counts)))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 1.0
        
        return normalized_entropy
    
    def _detect_recurrence(self, outcomes, embedding_dim=3, delay=1):
        """
        Detect recurrence in the time series using recurrence plots.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            embedding_dim: Embedding dimension
            delay: Time delay
            
        Returns:
            tuple: (recurrence_rate, determinism)
        """
        if len(outcomes) < embedding_dim * delay:
            return 0.0, 0.0
        
        # Convert to numeric
        numeric = np.array([1 if o == 'P' else 0 for o in outcomes])
        
        # Create embedded vectors
        vectors = []
        for i in range(len(numeric) - (embedding_dim - 1) * delay):
            vector = [numeric[i + j * delay] for j in range(embedding_dim)]
            vectors.append(vector)
        
        # Calculate distance matrix
        n_vectors = len(vectors)
        distance_matrix = np.zeros((n_vectors, n_vectors))
        
        for i in range(n_vectors):
            for j in range(i, n_vectors):
                # Euclidean distance
                dist = np.sqrt(sum((vectors[i][k] - vectors[j][k]) ** 2 for k in range(embedding_dim)))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Create recurrence matrix
        threshold = np.percentile(distance_matrix, 20)  # Use 20th percentile as threshold
        recurrence_matrix = distance_matrix < threshold
        
        # Calculate recurrence rate
        recurrence_rate = np.sum(recurrence_matrix) / (n_vectors ** 2)
        
        # Calculate determinism (ratio of diagonal lines)
        min_line_length = 2
        diag_lengths = []
        
        for i in range(-(n_vectors - min_line_length), n_vectors - min_line_length + 1):
            diag = np.diag(recurrence_matrix, k=i)
            
            # Find diagonal lines
            line_length = 0
            for point in diag:
                if point:
                    line_length += 1
                else:
                    if line_length >= min_line_length:
                        diag_lengths.append(line_length)
                    line_length = 0
            
            # Check last line
            if line_length >= min_line_length:
                diag_lengths.append(line_length)
        
        # Calculate determinism
        if np.sum(recurrence_matrix) > 0:
            determinism = sum(diag_lengths) / np.sum(recurrence_matrix)
        else:
            determinism = 0.0
        
        return recurrence_rate, determinism
    
    def _predict_from_transition_matrix(self, current_symbol):
        """
        Predict next outcome based on transition matrix.
        
        Args:
            current_symbol: Current symbol
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if current_symbol not in self.transition_matrix:
            return 'B'  # Default to Banker
        
        # Get transition probabilities
        transitions = self.transition_matrix[current_symbol]
        total = sum(transitions.values())
        
        if total == 0:
            return 'B'  # Default to Banker
        
        # Calculate probabilities for each possible next symbol
        p_prob = 0
        b_prob = 0
        
        for next_symbol, count in transitions.items():
            prob = count / total
            # Last bit of next symbol determines P or B
            if next_symbol[-1] == '1':  # P
                p_prob += prob
            else:  # B
                b_prob += prob
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        return 'P' if p_prob > b_prob else 'B'
    
    def _predict_from_forbidden_patterns(self, current_symbol):
        """
        Predict next outcome based on forbidden patterns.
        
        Args:
            current_symbol: Current symbol
            
        Returns:
            tuple: (prediction, confidence)
        """
        # Check possible next symbols
        possible_p = current_symbol[1:] + '1'  # Symbol if next is P
        possible_b = current_symbol[1:] + '0'  # Symbol if next is B
        
        # Check if either is a forbidden pattern
        p_forbidden = possible_p in self.forbidden_patterns
        b_forbidden = possible_b in self.forbidden_patterns
        
        if p_forbidden and not b_forbidden:
            return 'B', 1.0
        elif b_forbidden and not p_forbidden:
            return 'P', 1.0
        elif p_forbidden and b_forbidden:
            # Both forbidden - use transition matrix
            return self._predict_from_transition_matrix(current_symbol), 0.5
        else:
            # Neither forbidden - no strong prediction
            return None, 0.0
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using symbolic dynamics analysis.
        
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
        
        # Not enough data - use simple frequency analysis
        if len(filtered) < self.min_samples:
            p_count = filtered.count('P')
            b_count = filtered.count('B')
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'
        
        # Update memory with new outcome
        if len(filtered) > 0 and (len(self.outcome_memory) == 0 or filtered[-1] != self.outcome_memory[-1]):
            self.outcome_memory.append(filtered[-1])
        
        # Update symbol statistics
        self._update_symbol_statistics(filtered)
        
        # Update forbidden patterns
        self.forbidden_patterns = self._identify_forbidden_patterns()
        
        # Calculate entropy
        entropy = self._calculate_topological_entropy()
        
        # Detect recurrence
        recurrence_rate, determinism = self._detect_recurrence(filtered)
        
        logger.debug(f"Entropy: {entropy:.3f}, Recurrence: {recurrence_rate:.3f}, Determinism: {determinism:.3f}")
        
        # Make prediction based on current state
        if len(self.symbol_memory) > 0:
            current_symbol = self.symbol_memory[-1]
            
            # Try forbidden pattern prediction first
            forbidden_pred, confidence = self._predict_from_forbidden_patterns(current_symbol)
            
            if forbidden_pred and confidence > self.forbidden_pattern_weight:
                # Use forbidden pattern prediction
                prediction = forbidden_pred
            elif entropy < self.entropy_threshold or determinism > self.recurrence_threshold:
                # Low entropy or high determinism - use transition matrix
                prediction = self._predict_from_transition_matrix(current_symbol)
            else:
                # High entropy - use recent trend
                recent = filtered[-20:] if len(filtered) >= 20 else filtered
                p_count = recent.count('P')
                b_count = recent.count('B')
                
                # Apply banker bias
                b_count += b_count * self.banker_bias
                
                prediction = 'P' if p_count > b_count else 'B'
        else:
            # No symbols yet - use frequency analysis
            p_count = filtered.count('P')
            b_count = filtered.count('B')
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            prediction = 'P' if p_count > b_count else 'B'
        
        # Update performance tracking if we have actual outcome
        if len(self.outcome_memory) >= 2:
            # Get previous prediction (we don't store it, so recalculate)
            prev_outcomes = list(self.outcome_memory)[:-1]
            prev_filtered = [o for o in prev_outcomes if o in ['P', 'B']]
            
            if len(prev_filtered) >= self.min_samples and len(self.symbol_memory) >= 2:
                prev_symbol = self.symbol_memory[-2]
                
                # Recalculate previous prediction
                forbidden_pred, confidence = self._predict_from_forbidden_patterns(prev_symbol)
                
                if forbidden_pred and confidence > self.forbidden_pattern_weight:
                    prev_prediction = forbidden_pred
                elif entropy < self.entropy_threshold or determinism > self.recurrence_threshold:
                    prev_prediction = self._predict_from_transition_matrix(prev_symbol)
                else:
                    recent = prev_filtered[-20:] if len(prev_filtered) >= 20 else prev_filtered
                    p_count = recent.count('P')
                    b_count = recent.count('B')
                    b_count += b_count * self.banker_bias
                    prev_prediction = 'P' if p_count > b_count else 'B'
                
                # Check if prediction was correct
                actual = self.outcome_memory[-1]
                correct = prev_prediction == actual
                
                if correct:
                    self.correct_predictions += 1
                self.total_predictions += 1
        
        # Log performance
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logger.debug(f"Current accuracy: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")
        
        return prediction
