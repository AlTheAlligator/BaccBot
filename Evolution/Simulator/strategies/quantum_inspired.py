"""
Quantum-Inspired strategy implementation.
"""

import logging
import numpy as np
import math
from collections import defaultdict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class QuantumInspiredStrategy(BaseStrategy):
    """
    Quantum-Inspired strategy that applies quantum computing concepts to decision making.
    
    This strategy uses quantum-inspired algorithms like quantum amplitude amplification
    and superposition to explore multiple betting strategies simultaneously and
    collapse to the optimal decision based on observed outcomes.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Quantum-Inspired strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.num_qubits = params.get('num_qubits', 3)
        self.amplitude_boost = params.get('amplitude_boost', 1.5)
        self.interference_factor = params.get('interference_factor', 0.3)
        self.grover_iterations = params.get('grover_iterations', 2)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        
        # Initialize quantum-inspired state
        self.amplitudes = {'P': 0.5, 'B': 0.5}  # Initial equal superposition
        
        # Initialize sub-strategies
        self.strategies = {
            'frequency': {'weight': 1.0, 'amplitude': 0.0},
            'streak': {'weight': 1.0, 'amplitude': 0.0},
            'pattern': {'weight': 1.0, 'amplitude': 0.0},
            'alternating': {'weight': 1.0, 'amplitude': 0.0}
        }
        
        # Pattern store for pattern matching
        self.pattern_store = defaultdict(lambda: {'P': 0, 'B': 0})
        
        # Performance tracking
        self.strategy_performance = {
            'frequency': {'correct': 0, 'total': 0},
            'streak': {'correct': 0, 'total': 0},
            'pattern': {'correct': 0, 'total': 0},
            'alternating': {'correct': 0, 'total': 0}
        }
        
        # Last predictions
        self.last_predictions = {
            'frequency': None,
            'streak': None,
            'pattern': None,
            'alternating': None
        }
        
        logger.info(f"Initialized Quantum-Inspired strategy with num_qubits={self.num_qubits}")
    
    def _update_pattern_store(self, outcomes):
        """
        Update pattern store with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Process patterns of different lengths
        for length in range(1, self.num_qubits + 1):
            if len(outcomes) <= length:
                continue
                
            # Process each pattern
            for i in range(len(outcomes) - length):
                pattern = tuple(outcomes[i:i+length])
                next_outcome = outcomes[i+length]
                
                # Update pattern store
                self.pattern_store[pattern][next_outcome] += 1
    
    def _update_strategy_performance(self, actual_outcome):
        """
        Update performance metrics for each strategy.
        
        Args:
            actual_outcome: The actual outcome ('P' or 'B')
        """
        for strategy, prediction in self.last_predictions.items():
            if prediction is not None:
                self.strategy_performance[strategy]['total'] += 1
                if prediction == actual_outcome:
                    self.strategy_performance[strategy]['correct'] += 1
    
    def _update_amplitudes(self, outcomes):
        """
        Update quantum-inspired amplitudes based on observed outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Calculate strategy predictions and performance
        frequency_pred = self._frequency_analysis(outcomes)
        streak_pred = self._streak_analysis(outcomes)
        pattern_pred = self._pattern_analysis(outcomes)
        alternating_pred = self._alternating_analysis(outcomes)
        
        # Store predictions for performance tracking
        self.last_predictions = {
            'frequency': frequency_pred,
            'streak': streak_pred,
            'pattern': pattern_pred,
            'alternating': alternating_pred
        }
        
        # Calculate strategy weights based on performance
        total_strategies = len(self.strategies)
        for strategy, perf in self.strategy_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                # Scale to [0, 1] range with random baseline at 0.5
                scaled_accuracy = max(0, (accuracy - 0.5) * 2)
                self.strategies[strategy]['weight'] = 0.5 + scaled_accuracy * 0.5
            else:
                self.strategies[strategy]['weight'] = 1.0 / total_strategies
        
        # Normalize weights
        total_weight = sum(s['weight'] for s in self.strategies.values())
        for strategy in self.strategies:
            self.strategies[strategy]['weight'] /= total_weight
        
        # Calculate strategy amplitudes
        for strategy, pred in self.last_predictions.items():
            if pred == 'P':
                self.strategies[strategy]['amplitude'] = math.sqrt(self.strategies[strategy]['weight'])
            else:  # 'B'
                self.strategies[strategy]['amplitude'] = -math.sqrt(self.strategies[strategy]['weight'])
        
        # Calculate outcome amplitudes through interference
        p_amplitude = sum(s['amplitude'] for s, pred in zip(self.strategies.values(), self.last_predictions.values()) 
                          if pred == 'P')
        b_amplitude = sum(s['amplitude'] for s, pred in zip(self.strategies.values(), self.last_predictions.values()) 
                          if pred == 'B')
        
        # Normalize amplitudes
        norm = math.sqrt(p_amplitude**2 + b_amplitude**2)
        if norm > 0:
            self.amplitudes['P'] = p_amplitude / norm
            self.amplitudes['B'] = b_amplitude / norm
    
    def _amplify_amplitudes(self):
        """
        Apply quantum-inspired amplitude amplification (Grover's algorithm).
        """
        # Apply Grover iterations
        for _ in range(self.grover_iterations):
            # Phase inversion (oracle)
            if abs(self.amplitudes['P']) > abs(self.amplitudes['B']):
                self.amplitudes['P'] *= -1
            else:
                self.amplitudes['B'] *= -1
            
            # Diffusion (reflection about average)
            avg = (self.amplitudes['P'] + self.amplitudes['B']) / 2
            self.amplitudes['P'] = 2 * avg - self.amplitudes['P']
            self.amplitudes['B'] = 2 * avg - self.amplitudes['B']
            
            # Normalize
            norm = math.sqrt(self.amplitudes['P']**2 + self.amplitudes['B']**2)
            self.amplitudes['P'] /= norm
            self.amplitudes['B'] /= norm
    
    def _frequency_analysis(self, outcomes):
        """
        Perform frequency analysis on outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        p_count = outcomes.count('P')
        b_count = outcomes.count('B')
        
        return 'P' if p_count > b_count else 'B'
    
    def _streak_analysis(self, outcomes):
        """
        Analyze streaks in outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < 3:
            return 'B'  # Default to Banker
        
        # Find current streak
        current = outcomes[-1]
        streak_length = 1
        for i in range(len(outcomes) - 2, -1, -1):
            if outcomes[i] == current:
                streak_length += 1
            else:
                break
        
        # If streak is long, predict it will continue
        if streak_length >= 3:
            return current
        # Otherwise predict it will break
        else:
            return 'P' if current == 'B' else 'B'
    
    def _pattern_analysis(self, outcomes):
        """
        Analyze patterns in outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < self.num_qubits:
            return 'B'  # Default to Banker
        
        # Get current pattern
        pattern = tuple(outcomes[-self.num_qubits:])
        
        # Check if pattern exists in store
        if pattern in self.pattern_store:
            counts = self.pattern_store[pattern]
            p_count = counts['P']
            b_count = counts['B']
            
            if p_count + b_count > 0:
                return 'P' if p_count > b_count else 'B'
        
        return 'B'  # Default to Banker
    
    def _alternating_analysis(self, outcomes):
        """
        Analyze alternating patterns in outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < 4:
            return 'B'  # Default to Banker
        
        # Check for alternating pattern
        alternating_count = 0
        for i in range(len(outcomes) - 3, len(outcomes) - 1):
            if outcomes[i] != outcomes[i + 1]:
                alternating_count += 1
        
        # If recent outcomes are alternating, predict the opposite of last outcome
        if alternating_count >= 2:
            return 'P' if outcomes[-1] == 'B' else 'B'
        else:
            return outcomes[-1]  # Predict same as last outcome
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using quantum-inspired algorithms.
        
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
        
        # Update pattern store
        self._update_pattern_store(filtered)
        
        # Update quantum-inspired state based on observed outcomes
        self._update_amplitudes(filtered)
        
        # Apply quantum-inspired amplitude amplification
        self._amplify_amplitudes()
        
        # Apply banker bias
        self.amplitudes['B'] += self.banker_bias
        
        # Normalize amplitudes
        total = abs(self.amplitudes['P']) + abs(self.amplitudes['B'])
        self.amplitudes['P'] /= total
        self.amplitudes['B'] /= total
        
        # Collapse quantum state to classical decision
        p_prob = self.amplitudes['P']**2
        b_prob = self.amplitudes['B']**2
        
        logger.debug(f"Quantum amplitudes: P={self.amplitudes['P']:.3f}, B={self.amplitudes['B']:.3f}")
        logger.debug(f"Quantum probabilities: P={p_prob:.3f}, B={b_prob:.3f}")
        
        return 'P' if p_prob > b_prob else 'B'
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        strategy_accuracies = {}
        for strategy, perf in self.strategy_performance.items():
            if perf['total'] > 0:
                strategy_accuracies[strategy] = perf['correct'] / perf['total']
            else:
                strategy_accuracies[strategy] = 0.0
                
        return {
            "amplitudes": {k: float(v) for k, v in self.amplitudes.items()},
            "strategy_weights": {k: float(s['weight']) for k, s in self.strategies.items()},
            "strategy_accuracies": strategy_accuracies,
            "pattern_count": len(self.pattern_store)
        }
