import logging
import numpy as np
from collections import defaultdict, deque

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BayesianInferenceStrategy(BaseStrategy):
    """
    A strategy that uses Bayesian inference to update outcome probabilities.
    
    This strategy maintains probability distributions over outcomes and updates them
    using Bayes' rule as new evidence is observed. It also considers contextual
    features to condition the probability estimates.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Bayesian Inference strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.min_samples = params.get('min_samples', 15)  # Minimum samples before making predictions
        self.confidence_threshold = params.get('confidence_threshold', 0.55)  # Min confidence to place bet
        self.prior_strength = params.get('prior_strength', 3.0)  # Strength of prior beliefs
        self.pattern_length = params.get('pattern_length', 3)  # Length of patterns to consider
        self.banker_edge_adjustment = params.get('banker_edge_adjustment', 0.011)  # Adjustment for banker edge
        self.use_recency_weighting = params.get('use_recency_weighting', True)  # Whether to weight recent outcomes more
        self.recency_factor = params.get('recency_factor', 0.95)  # Factor for recency weighting
        self.use_context_features = params.get('use_context_features', True)  # Whether to use contextual features
        self.feature_window = params.get('feature_window', 10)  # Window for feature calculation
        
        # Initialize probability distributions
        self.base_probs = {
            'P': {'alpha': self.prior_strength, 'beta': self.prior_strength},
            'B': {'alpha': self.prior_strength, 'beta': self.prior_strength}
        }
        
        # Context-specific distributions
        self.context_probs = defaultdict(
            lambda: {
                'P': {'alpha': self.prior_strength / 2, 'beta': self.prior_strength / 2},
                'B': {'alpha': self.prior_strength / 2, 'beta': self.prior_strength / 2}
            }
        )
        
        # Feature tracking
        self.feature_history = deque(maxlen=100)
        self.last_features = None
        
        logger.info(f"Initialized Bayesian Inference strategy with prior_strength={self.prior_strength}, "
                   f"confidence_threshold={self.confidence_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on Bayesian probability estimates.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable inference
            logger.debug(f"Not enough data for Bayesian inference ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update probability distributions with new evidence
        self._update_distributions(non_tie_outcomes)
        
        # Extract current features
        features = self._extract_features(non_tie_outcomes)
        self.last_features = features
        
        # Get probability estimates
        p_prob, b_prob, confidence = self._estimate_probabilities(features)
        
        # Apply banker edge adjustment
        b_prob += self.banker_edge_adjustment
        
        # Normalize probabilities
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        logger.debug(f"Bayesian probabilities: P={p_prob:.3f}, B={b_prob:.3f}, confidence={confidence:.3f}")
        
        # Make decision based on probabilities and confidence
        if confidence >= self.confidence_threshold:
            if p_prob > b_prob:
                return "P"
            else:
                return "B"
        else:
            logger.debug(f"Confidence too low: {confidence:.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _update_distributions(self, outcomes):
        """
        Update probability distributions with new evidence.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update base distributions
        if self.use_recency_weighting:
            # Apply recency weighting to updates
            for i, outcome in enumerate(outcomes):
                # Calculate weight based on recency
                weight = self.recency_factor ** (len(outcomes) - i - 1)
                
                # Update distributions for both outcomes
                for side in ['P', 'B']:
                    if outcome == side:
                        self.base_probs[side]['alpha'] += weight
                    else:
                        self.base_probs[side]['beta'] += weight
        else:
            # Simple counting without weighting
            for outcome in outcomes:
                for side in ['P', 'B']:
                    if outcome == side:
                        self.base_probs[side]['alpha'] += 1
                    else:
                        self.base_probs[side]['beta'] += 1
        
        # Update context-specific distributions if enabled
        if self.use_context_features:
            for i in range(self.pattern_length, len(outcomes)):
                # Get context (previous pattern)
                context = tuple(outcomes[i-self.pattern_length:i])
                outcome = outcomes[i]
                
                # Update distribution for this context
                for side in ['P', 'B']:
                    if outcome == side:
                        self.context_probs[context][side]['alpha'] += 1
                    else:
                        self.context_probs[context][side]['beta'] += 1
    
    def _extract_features(self, outcomes):
        """
        Extract features from recent outcomes for contextual probability estimation.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            dict: Features extracted from outcomes
        """
        if len(outcomes) < self.feature_window:
            return {}
        
        # Get recent outcomes for feature calculation
        recent = outcomes[-self.feature_window:]
        
        features = {}
        
        # Calculate basic statistics
        p_count = recent.count('P')
        b_count = recent.count('B')
        total = p_count + b_count
        
        features['p_ratio'] = p_count / total if total > 0 else 0.5
        features['recent_bias'] = abs(p_count - b_count) / total if total > 0 else 0
        
        # Get current pattern
        if len(outcomes) >= self.pattern_length:
            features['current_pattern'] = tuple(outcomes[-self.pattern_length:])
        
        # Detect streak
        streak_value, streak_length = self._get_current_streak(outcomes)
        features['streak_value'] = streak_value
        features['streak_length'] = streak_length
        
        # Detect alternating pattern
        is_alternating, last_value = self._is_alternating(outcomes[-4:])
        features['is_alternating'] = is_alternating
        
        # Store features for debugging
        self.feature_history.append(features)
        
        return features
    
    def _estimate_probabilities(self, features):
        """
        Estimate probabilities using Bayesian inference with features.
        
        Args:
            features: Dict of current features
            
        Returns:
            tuple: (p_prob, b_prob, confidence)
        """
        # Get base probabilities
        p_base = self.base_probs['P']['alpha'] / (self.base_probs['P']['alpha'] + self.base_probs['P']['beta'])
        b_base = self.base_probs['B']['alpha'] / (self.base_probs['B']['alpha'] + self.base_probs['B']['beta'])
        
        # Initialize final probabilities with base probabilities
        p_prob = p_base
        b_prob = b_base
        
        if self.use_context_features and features:
            # Apply context-specific adjustments
            if 'current_pattern' in features:
                pattern = features['current_pattern']
                if pattern in self.context_probs:
                    # Get context-specific probabilities
                    context_dist_p = self.context_probs[pattern]['P']
                    context_dist_b = self.context_probs[pattern]['B']
                    
                    # Calculate probabilities from context distributions
                    total_p = context_dist_p['alpha'] + context_dist_p['beta']
                    total_b = context_dist_b['alpha'] + context_dist_b['beta']
                    
                    if total_p >= 3 and total_b >= 3:  # Minimum observations threshold
                        p_context = context_dist_p['alpha'] / total_p
                        b_context = context_dist_b['alpha'] / total_b
                        
                        # Blend base and context probabilities
                        context_weight = min(0.7, (total_p + total_b) / 30)  # Cap context influence
                        p_prob = p_prob * (1 - context_weight) + p_context * context_weight
                        b_prob = b_prob * (1 - context_weight) + b_context * context_weight
            
            # Apply streak-based adjustments
            if 'streak_length' in features and features['streak_length'] > 0:
                streak_length = features['streak_length']
                streak_value = features['streak_value']
                
                # Calculate streak influence
                streak_factor = min(0.3, streak_length * 0.05)  # Cap at 0.3
                
                if streak_value == 'P':
                    # Favor continuation but with diminishing returns
                    p_prob = p_prob * (1 + streak_factor)
                elif streak_value == 'B':
                    b_prob = b_prob * (1 + streak_factor)
            
            # Apply alternating pattern adjustment
            if 'is_alternating' in features and features['is_alternating']:
                last_value = features['current_pattern'][-1] if 'current_pattern' in features else None
                if last_value:
                    # Favor alternation
                    if last_value == 'P':
                        b_prob *= 1.2
                    else:
                        p_prob *= 1.2
        
        # Calculate confidence based on probability difference and evidence strength
        prob_diff = abs(p_prob - b_prob)
        base_evidence = min(sum(self.base_probs['P'].values()), sum(self.base_probs['B'].values()))
        evidence_factor = min(1.0, base_evidence / (self.prior_strength * 10))
        
        confidence = prob_diff * evidence_factor
        
        return p_prob, b_prob, confidence
    
    def get_distribution_stats(self):
        """Get current probability distribution statistics for debugging."""
        stats = {
            "base_distributions": {
                'P': dict(self.base_probs['P']),
                'B': dict(self.base_probs['B'])
            },
            "context_count": len(self.context_probs),
            "recent_features": list(self.feature_history)[-5:] if self.feature_history else [],
            "last_features": self.last_features
        }
        
        # Add top context patterns
        top_contexts = []
        for pattern, dist in self.context_probs.items():
            p_strength = dist['P']['alpha'] + dist['P']['beta']
            b_strength = dist['B']['alpha'] + dist['B']['beta']
            if p_strength + b_strength >= 5:  # Minimum observations threshold
                p_prob = dist['P']['alpha'] / (dist['P']['alpha'] + dist['P']['beta'])
                b_prob = dist['B']['alpha'] / (dist['B']['alpha'] + dist['B']['beta'])
                skew = abs(p_prob - b_prob)
                top_contexts.append((pattern, {'P': p_prob, 'B': b_prob, 'skew': skew}))
        
        # Sort by skew (more predictive patterns first)
        top_contexts.sort(key=lambda x: x[1]['skew'], reverse=True)
        stats["top_contexts"] = top_contexts[:5]
        
        return stats