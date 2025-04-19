"""
Conditional Probability Chain Strategy

This strategy uses Bayesian updating and conditional probabilities to predict
the next outcome based on chains of previous outcomes.
"""

import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ConditionalProbabilityChainStrategy:
    """
    A strategy that uses conditional probability chains and Bayesian updating
    to predict baccarat outcomes.
    
    Features:
    - Markov chain modeling of outcome transitions
    - Bayesian probability updating
    - Variable-length condition chains
    - Prior probability adjustment
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Conditional Probability Chain strategy.
        
        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}
        
        # Core parameters
        self.min_chain_length = params.get('min_chain_length', 1)
        self.max_chain_length = params.get('max_chain_length', 4)
        self.min_samples = params.get('min_samples', 5)
        self.confidence_threshold = params.get('confidence_threshold', 0.55)
        self.banker_bias = params.get('banker_bias', 0.01)
        
        # Advanced parameters
        self.use_bayesian_updating = params.get('use_bayesian_updating', True)
        self.prior_weight = params.get('prior_weight', 10)  # Weight of prior beliefs
        self.recency_factor = params.get('recency_factor', 0.95)  # For recency weighting
        self.use_variable_chains = params.get('use_variable_chains', True)
        self.chain_weight_factor = params.get('chain_weight_factor', 1.5)  # Longer chains get more weight
        
        # Initialize transition matrices for different chain lengths
        self.transitions = {}
        for length in range(self.min_chain_length, self.max_chain_length + 1):
            self.transitions[length] = defaultdict(lambda: {'P': 0, 'B': 0, 'total': 0})
            
        # Prior probabilities (based on house edge)
        self.prior_p_prob = 0.4462 / (0.4462 + 0.4585)  # Normalized for non-tie outcomes
        self.prior_b_prob = 0.4585 / (0.4462 + 0.4585)
        
        # State tracking
        self.outcome_history = []
        
    def _update_transitions(self, outcomes: List[str]):
        """
        Update transition probabilities based on observed outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
        """
        # Filter out ties
        filtered = [o for o in outcomes if o in ['P', 'B']]
        
        if len(filtered) < self.min_chain_length + 1:
            return
            
        # Update transition matrices for each chain length
        for length in range(self.min_chain_length, self.max_chain_length + 1):
            if len(filtered) < length + 1:
                continue
                
            # Update transitions for each chain in the history
            for i in range(len(filtered) - length):
                chain = ''.join(filtered[i:i+length])
                next_outcome = filtered[i+length]
                
                # Apply recency weighting to older observations
                recency_weight = self.recency_factor ** (len(filtered) - i - length - 1)
                
                # Update counts
                self.transitions[length][chain][next_outcome] += recency_weight
                self.transitions[length][chain]['total'] += recency_weight
    
    def _get_chain_prediction(self, chain: str, length: int) -> Tuple[Dict[str, float], float]:
        """
        Get prediction based on a specific chain.
        
        Args:
            chain: The chain string
            length: Length of the chain
            
        Returns:
            tuple: (probabilities, confidence)
        """
        stats = self.transitions[length][chain]
        
        if stats['total'] < self.min_samples:
            # Not enough samples, use prior probabilities
            return {'P': self.prior_p_prob, 'B': self.prior_b_prob}, 0.0
            
        # Calculate probabilities
        p_count = stats['P']
        b_count = stats['B']
        total = stats['total']
        
        # Apply banker bias
        b_count += b_count * self.banker_bias
        
        # Calculate probabilities
        p_prob = p_count / total if total > 0 else self.prior_p_prob
        b_prob = b_count / total if total > 0 else self.prior_b_prob
        
        # Normalize
        total_prob = p_prob + b_prob
        p_prob /= total_prob
        b_prob /= total_prob
        
        # Calculate confidence based on sample size and probability difference
        confidence = min(1.0, stats['total'] / 50.0) * abs(p_prob - b_prob) * 2
        
        return {'P': p_prob, 'B': b_prob}, confidence
    
    def _apply_bayesian_update(self, predictions: List[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
        """
        Apply Bayesian updating to combine predictions from different chain lengths.
        
        Args:
            predictions: List of (probabilities, confidence) tuples
            
        Returns:
            dict: Updated probabilities
        """
        if not predictions:
            return {'P': self.prior_p_prob, 'B': self.prior_b_prob}
            
        if not self.use_bayesian_updating:
            # Simple weighted average
            p_prob = sum(probs['P'] * conf for probs, conf in predictions) / sum(conf for _, conf in predictions) if sum(conf for _, conf in predictions) > 0 else self.prior_p_prob
            b_prob = sum(probs['B'] * conf for probs, conf in predictions) / sum(conf for _, conf in predictions) if sum(conf for _, conf in predictions) > 0 else self.prior_b_prob
            
            # Normalize
            total = p_prob + b_prob
            return {'P': p_prob / total, 'B': b_prob / total}
        
        # Start with prior probabilities
        p_posterior = self.prior_p_prob
        b_posterior = self.prior_b_prob
        
        # Apply Bayes' rule sequentially for each prediction
        for probs, conf in predictions:
            if conf <= 0:
                continue
                
            # Weight the likelihood by confidence
            p_likelihood = probs['P'] ** conf
            b_likelihood = probs['B'] ** conf
            
            # Calculate posterior
            p_unnormalized = p_posterior * p_likelihood
            b_unnormalized = b_posterior * b_likelihood
            
            # Normalize
            total = p_unnormalized + b_unnormalized
            if total > 0:
                p_posterior = p_unnormalized / total
                b_posterior = b_unnormalized / total
        
        return {'P': p_posterior, 'B': b_posterior}
    
    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using conditional probability chains.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Always start from game 7
        if len(outcomes) < 7:
            return 'B'  # Default to Banker for initial games
            
        # Store outcome history
        self.outcome_history = outcomes.copy()
        
        # Update transition probabilities
        self._update_transitions(outcomes)
        
        # Filter out ties
        filtered = [o for o in outcomes if o in ['P', 'B']]
        
        if len(filtered) < self.min_chain_length:
            return 'B'  # Not enough data, default to Banker
        
        # Get predictions for each chain length
        predictions = []
        
        for length in range(self.min_chain_length, self.max_chain_length + 1):
            if len(filtered) < length:
                continue
                
            # Get current chain
            chain = ''.join(filtered[-length:])
            
            # Get prediction for this chain
            probs, confidence = self._get_chain_prediction(chain, length)
            
            # Weight longer chains more heavily if enabled
            if self.use_variable_chains:
                confidence *= (length / self.min_chain_length) ** self.chain_weight_factor
                
            predictions.append((probs, confidence))
        
        # Apply Bayesian updating
        final_probs = self._apply_bayesian_update(predictions)
        
        # Make decision
        if final_probs['P'] > final_probs['B'] and final_probs['P'] > self.confidence_threshold:
            return 'P'
        elif final_probs['B'] > final_probs['P'] and final_probs['B'] > self.confidence_threshold:
            return 'B'
        else:
            # Not confident enough, default to Banker
            return 'B'
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.
        
        Returns:
            dict: Strategy statistics
        """
        # Find chains with strong predictive power
        strong_chains = {}
        
        for length, chains in self.transitions.items():
            for chain, stats in chains.items():
                if stats['total'] >= self.min_samples:
                    p_prob = stats['P'] / stats['total']
                    b_prob = stats['B'] / stats['total']
                    max_prob = max(p_prob, b_prob)
                    
                    if max_prob > 0.65:  # Only include strong predictors
                        strong_chains[f"{chain} (len={length})"] = {
                            'P_next': f"{p_prob:.2f}",
                            'B_next': f"{b_prob:.2f}",
                            'samples': stats['total']
                        }
        
        # Get current chains
        current_chains = {}
        filtered = [o for o in self.outcome_history if o in ['P', 'B']]
        
        for length in range(self.min_chain_length, self.max_chain_length + 1):
            if len(filtered) >= length:
                chain = ''.join(filtered[-length:])
                stats = self.transitions[length][chain]
                
                if stats['total'] > 0:
                    p_prob = stats['P'] / stats['total']
                    b_prob = stats['B'] / stats['total']
                    
                    current_chains[f"Current {length}-chain"] = {
                        'chain': chain,
                        'P_prob': f"{p_prob:.2f}",
                        'B_prob': f"{b_prob:.2f}",
                        'samples': stats['total']
                    }
        
        return {
            "strategy": "Conditional Probability Chain",
            "strong_chains": strong_chains,
            "current_chains": current_chains,
            "total_chains_tracked": sum(len(chains) for chains in self.transitions.values())
        }
