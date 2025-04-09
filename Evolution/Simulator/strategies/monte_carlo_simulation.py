"""
Monte Carlo Simulation strategy implementation.
"""

import logging
import numpy as np
import random
from collections import defaultdict, Counter
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MonteCarloSimulationStrategy(BaseStrategy):
    """
    Monte Carlo Simulation strategy for baccarat betting.
    
    This strategy runs multiple simulations based on historical patterns to
    determine the most likely next outcome and the optimal betting decision.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Monte Carlo Simulation strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.num_simulations = params.get('num_simulations', 1000)  # Number of simulations to run
        self.min_samples = params.get('min_samples', 20)  # Minimum samples before making predictions
        self.pattern_length = params.get('pattern_length', 3)  # Length of patterns to consider
        self.confidence_threshold = params.get('confidence_threshold', 0.55)  # Threshold for making a bet
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.use_transition_matrix = params.get('use_transition_matrix', True)  # Whether to use transition matrix
        self.use_pattern_matching = params.get('use_pattern_matching', True)  # Whether to use pattern matching
        
        # For tracking state
        self.transition_matrix = defaultdict(lambda: {'P': 0, 'B': 0})
        self.pattern_frequencies = defaultdict(lambda: {'P': 0, 'B': 0})
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
        logger.info(f"Initialized Monte Carlo Simulation strategy with num_simulations={self.num_simulations}")
    
    def _update_transition_matrix(self, outcomes):
        """
        Update the transition matrix with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        if len(outcomes) < 2:
            return
        
        # Update first-order transitions
        for i in range(len(outcomes) - 1):
            current = outcomes[i]
            next_outcome = outcomes[i + 1]
            self.transition_matrix[current][next_outcome] += 1
        
        # Update pattern-based transitions
        if len(outcomes) >= self.pattern_length + 1:
            for i in range(len(outcomes) - self.pattern_length):
                pattern = tuple(outcomes[i:i+self.pattern_length])
                next_outcome = outcomes[i+self.pattern_length]
                self.pattern_frequencies[pattern][next_outcome] += 1
    
    def _get_transition_probabilities(self, state):
        """
        Get transition probabilities for a given state.
        
        Args:
            state: Current state (outcome or pattern)
            
        Returns:
            tuple: (p_prob, b_prob) - Transition probabilities
        """
        if isinstance(state, tuple) and state in self.pattern_frequencies:
            # Pattern-based transition
            counts = self.pattern_frequencies[state]
            total = counts['P'] + counts['B']
            
            if total > 0:
                p_prob = counts['P'] / total
                b_prob = counts['B'] / total
                return p_prob, b_prob
        
        if isinstance(state, str) and state in self.transition_matrix:
            # First-order transition
            counts = self.transition_matrix[state]
            total = counts['P'] + counts['B']
            
            if total > 0:
                p_prob = counts['P'] / total
                b_prob = counts['B'] / total
                return p_prob, b_prob
        
        # Default to global frequencies
        p_count = sum(1 for o in self.outcomes if o == 'P')
        b_count = sum(1 for o in self.outcomes if o == 'B')
        total = p_count + b_count
        
        if total > 0:
            p_prob = p_count / total
            b_prob = b_count / total
            return p_prob, b_prob
        
        # Equal probability if no data
        return 0.5, 0.5
    
    def _run_simulation(self, initial_state):
        """
        Run a single Monte Carlo simulation.
        
        Args:
            initial_state: Initial state for the simulation
            
        Returns:
            str: Simulated next outcome ('P' or 'B')
        """
        current_state = initial_state
        
        # Get transition probabilities
        if self.use_pattern_matching and isinstance(current_state, tuple):
            p_prob, b_prob = self._get_transition_probabilities(current_state)
        elif self.use_transition_matrix and isinstance(current_state, str):
            p_prob, b_prob = self._get_transition_probabilities(current_state)
        else:
            # Use global frequencies
            p_prob, b_prob = self._get_transition_probabilities(None)
        
        # Generate next outcome based on probabilities
        if random.random() < p_prob:
            return 'P'
        else:
            return 'B'
    
    def _run_monte_carlo_simulations(self, outcomes):
        """
        Run multiple Monte Carlo simulations to predict the next outcome.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: (prediction, confidence)
        """
        # Determine initial state
        if self.use_pattern_matching and len(outcomes) >= self.pattern_length:
            initial_state = tuple(outcomes[-self.pattern_length:])
        elif self.use_transition_matrix and len(outcomes) > 0:
            initial_state = outcomes[-1]
        else:
            initial_state = None
        
        # Run simulations
        simulation_results = []
        for _ in range(self.num_simulations):
            next_outcome = self._run_simulation(initial_state)
            simulation_results.append(next_outcome)
        
        # Count results
        result_counts = Counter(simulation_results)
        total_simulations = len(simulation_results)
        
        # Calculate probabilities
        p_prob = result_counts.get('P', 0) / total_simulations
        b_prob = result_counts.get('B', 0) / total_simulations
        
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
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using Monte Carlo simulations.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)
        
        # Filter out ties
        filtered_outcomes = [o for o in outcomes if o in ['P', 'B']]
        
        # Store for later use
        self.outcomes = filtered_outcomes
        
        # Not enough data - default to Banker (slight edge due to commission)
        if len(filtered_outcomes) < self.min_samples:
            return 'B'
        
        # Update transition matrix and pattern frequencies
        self._update_transition_matrix(filtered_outcomes)
        
        # Run Monte Carlo simulations
        bet, confidence = self._run_monte_carlo_simulations(filtered_outcomes)
        
        # Always return a bet (no skipping)
        return bet
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "num_simulations": self.num_simulations,
            "transition_matrix_size": len(self.transition_matrix),
            "pattern_frequencies_size": len(self.pattern_frequencies),
            "pattern_length": self.pattern_length
        }
