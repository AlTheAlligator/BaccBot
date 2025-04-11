"""
Bayesian Network strategy implementation.
"""

import logging
import numpy as np
from collections import defaultdict, deque
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BayesianNetworkStrategy(BaseStrategy):
    """
    Bayesian Network strategy that models conditional dependencies between outcomes.
    
    This strategy builds a dynamic Bayesian network that updates with each new outcome,
    models dependencies between consecutive outcomes and longer patterns, and uses
    Bayesian inference to calculate the probability of each outcome.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Bayesian Network strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.max_parents = params.get('max_parents', 3)
        self.prior_strength = params.get('prior_strength', 1.0)
        self.confidence_threshold = params.get('confidence_threshold', 0.55)
        self.learning_rate = params.get('learning_rate', 0.1)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.memory_length = params.get('memory_length', 200)
        
        # Initialize Bayesian network
        self.conditional_probs = {}
        for i in range(1, self.max_parents + 1):
            self.conditional_probs[i] = defaultdict(lambda: {'P': self.prior_strength, 'B': self.prior_strength})
        
        # Initialize memory
        self.outcome_memory = deque(maxlen=self.memory_length)
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Parent set scores
        self.parent_set_scores = np.ones(self.max_parents) / self.max_parents
    
    def _update_conditional_probs(self, outcomes):
        """
        Update conditional probabilities based on observed outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        if len(outcomes) <= self.max_parents:
            return
        
        # Update conditional probabilities for each parent set size
        for parent_size in range(1, self.max_parents + 1):
            if len(outcomes) > parent_size:
                # Get parent values (previous outcomes)
                parents = tuple(outcomes[-(parent_size+1):-1])
                # Get current outcome
                outcome = outcomes[-1]
                
                # Update count for this parent configuration
                self.conditional_probs[parent_size][parents][outcome] += 1
    
    def _calculate_conditional_prob(self, parents, outcome, parent_size):
        """
        Calculate conditional probability P(outcome | parents).
        
        Args:
            parents: Tuple of parent values
            outcome: Outcome to calculate probability for ('P' or 'B')
            parent_size: Number of parents
            
        Returns:
            float: Conditional probability
        """
        if parent_size not in self.conditional_probs or parents not in self.conditional_probs[parent_size]:
            # No data for this parent configuration - use prior
            return 0.5 - (0.01 if outcome == 'P' else -0.01)  # Slight banker bias in prior
        
        # Get counts
        counts = self.conditional_probs[parent_size][parents]
        p_count = counts['P']
        b_count = counts['B']
        total = p_count + b_count
        
        # Calculate probability with Laplace smoothing
        if outcome == 'P':
            return p_count / total
        else:  # 'B'
            return b_count / total
    
    def _update_parent_set_scores(self, prediction_results):
        """
        Update scores for each parent set size based on prediction accuracy.
        
        Args:
            prediction_results: Dictionary mapping parent size to (prediction, correct) tuples
        """
        for parent_size, (_, correct) in prediction_results.items():
            idx = parent_size - 1  # Convert to 0-based index
            if correct:
                # Increase weight if prediction was correct
                self.parent_set_scores[idx] += self.learning_rate * (1 - self.parent_set_scores[idx])
            else:
                # Decrease weight if prediction was wrong
                self.parent_set_scores[idx] -= self.learning_rate * self.parent_set_scores[idx]
        
        # Ensure weights are positive
        self.parent_set_scores = np.maximum(0.01, self.parent_set_scores)
        # Normalize weights
        self.parent_set_scores = self.parent_set_scores / np.sum(self.parent_set_scores)
    
    def _predict_with_parent_set(self, outcomes, parent_size):
        """
        Make prediction using a specific parent set size.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            parent_size: Number of parents to use
            
        Returns:
            tuple: (prediction, confidence)
        """
        if len(outcomes) <= parent_size:
            return 'B', 0.5  # Default to Banker with neutral confidence
        
        # Get parent values
        parents = tuple(outcomes[-parent_size:])
        
        # Calculate conditional probabilities
        p_prob = self._calculate_conditional_prob(parents, 'P', parent_size)
        b_prob = self._calculate_conditional_prob(parents, 'B', parent_size)
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        # Determine prediction and confidence
        if p_prob > b_prob:
            return 'P', p_prob
        else:
            return 'B', b_prob
    
    def _combine_predictions(self, predictions):
        """
        Combine predictions from different parent set sizes.
        
        Args:
            predictions: Dictionary mapping parent size to (prediction, confidence) tuples
            
        Returns:
            str: Final prediction ('P' or 'B')
        """
        # Calculate weighted probabilities
        p_prob = 0
        b_prob = 0
        
        for parent_size, (prediction, confidence) in predictions.items():
            weight = self.parent_set_scores[parent_size - 1]  # Convert to 0-based index
            if prediction == 'P':
                p_prob += weight * confidence
            else:  # 'B'
                b_prob += weight * confidence
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        logger.debug(f"Combined probabilities: P={p_prob:.3f}, B={b_prob:.3f}")
        
        # Return prediction
        return 'P' if p_prob > b_prob else 'B'
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using Bayesian network analysis.
        
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
            # Update conditional probabilities
            self._update_conditional_probs(list(self.outcome_memory))
        
        # Make predictions with each parent set size
        predictions = {}
        for parent_size in range(1, self.max_parents + 1):
            prediction, confidence = self._predict_with_parent_set(filtered, parent_size)
            predictions[parent_size] = (prediction, confidence)
        
        # Log individual predictions
        for parent_size, (prediction, confidence) in predictions.items():
            logger.debug(f"Parent size {parent_size}: {prediction} with confidence {confidence:.3f}")
        
        # Combine predictions
        final_prediction = self._combine_predictions(predictions)
        
        # Update parent set scores if we have the actual outcome
        if len(self.outcome_memory) >= 2:
            # Get previous predictions
            prev_outcomes = list(self.outcome_memory)[:-1]
            prev_filtered = [o for o in prev_outcomes if o in ['P', 'B']]
            
            if len(prev_filtered) >= self.min_samples:
                # Calculate what each parent set would have predicted
                prediction_results = {}
                for parent_size in range(1, self.max_parents + 1):
                    prev_pred, _ = self._predict_with_parent_set(prev_filtered, parent_size)
                    actual = self.outcome_memory[-1]
                    correct = prev_pred == actual
                    prediction_results[parent_size] = (prev_pred, correct)
                
                # Update parent set scores
                self._update_parent_set_scores(prediction_results)
                
                # Update overall performance tracking
                if len(prediction_results) > 0:
                    # Use the combined prediction for overall tracking
                    prev_predictions = {ps: (pred, 0.5) for ps, (pred, _) in prediction_results.items()}
                    prev_combined = self._combine_predictions(prev_predictions)
                    actual = self.outcome_memory[-1]
                    correct = prev_combined == actual
                    
                    if correct:
                        self.correct_predictions += 1
                    self.total_predictions += 1
        
        # Log performance and weights
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logger.debug(f"Current accuracy: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")
            logger.debug(f"Parent set weights: {self.parent_set_scores}")
        
        return final_prediction
