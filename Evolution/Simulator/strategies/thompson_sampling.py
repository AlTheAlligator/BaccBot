import logging
from collections import defaultdict
from scipy.stats import beta

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ThompsonSamplingStrategy(BaseStrategy):
    """
    A strategy that uses Thompson Sampling to make betting decisions.
    
    Thompson Sampling is a Bayesian approach to the multi-armed bandit problem where
    we maintain probability distributions over the win rates for each bet option
    and make decisions by sampling from these distributions.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Thompson Sampling strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.prior_alpha = params.get('prior_alpha', 1.0)  # Prior alpha for Beta distribution
        self.prior_beta = params.get('prior_beta', 1.0)  # Prior beta for Beta distribution
        self.min_samples = params.get('min_samples', 15)  # Minimum samples before making predictions
        self.confidence_threshold = params.get('confidence_threshold', 0.56)  # Min confidence to place bet
        self.use_context = params.get('use_context', True)  # Whether to use context-aware sampling
        self.context_length = params.get('context_length', 3)  # Length of context pattern
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.exploration_weight = params.get('exploration_weight', 1.0)  # Weight for exploration vs exploitation
        self.min_context_observations = params.get('min_context_observations', 3)  # Minimum observations before using context
        self.discount_factor = params.get('discount_factor', 0.99)  # Factor to discount older observations
        self.use_recency_weighting = params.get('use_recency_weighting', True)  # Whether to weight recent observations more
        
        # Initialize distributions
        # Global distribution (no context)
        self.global_dist = {
            'P': {'alpha': self.prior_alpha, 'beta': self.prior_beta},
            'B': {'alpha': self.prior_alpha, 'beta': self.prior_beta}
        }
        
        # Context-specific distributions
        self.context_dists = defaultdict(
            lambda: {
                'P': {'alpha': self.prior_alpha, 'beta': self.prior_beta},
                'B': {'alpha': self.prior_alpha, 'beta': self.prior_beta}
            }
        )
        
        # Record of recent samples for debugging
        self.recent_samples = []
        
        logger.info(f"Initialized Thompson Sampling strategy with use_context={self.use_context}, "
                   f"confidence_threshold={self.confidence_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet using Thompson Sampling.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable sampling
            logger.debug(f"Not enough data for Thompson sampling ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update distributions with observed data
        self._update_distributions(non_tie_outcomes)
        
        # Get current context if using context-aware sampling
        context = None
        if self.use_context:
            context = self._get_context(non_tie_outcomes)
        
        # Sample from distributions
        p_sample = self._sample_win_probability(context, 'P')
        b_sample = self._sample_win_probability(context, 'B')
        
        # Apply banker bias
        b_sample += self.banker_bias
        
        # Record samples for tracking
        self.recent_samples.append((p_sample, b_sample, str(context)))
        
        # Calculate difference as a measure of confidence
        diff = abs(p_sample - b_sample)
        logger.debug(f"Thompson samples: P={p_sample:.3f}, B={b_sample:.3f}, diff={diff:.3f}")
        
        # Make decision based on sampled probabilities and confidence threshold
        if diff >= self.confidence_threshold:
            if p_sample > b_sample:
                return "P"
            else:
                return "B"
        else:
            logger.debug(f"Difference too low: {diff:.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _get_context(self, outcomes):
        """
        Get the current context based on recent outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            tuple: Context tuple of recent outcomes
        """
        if len(outcomes) < self.context_length:
            return None
            
        # Use the most recent N outcomes as context
        return tuple(outcomes[-self.context_length:])
    
    def _update_distributions(self, outcomes):
        """
        Update Beta distributions with observed transitions.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update global distribution (higher weight on more recent outcomes)
        if self.use_recency_weighting:
            # Apply weighting to prioritize recent outcomes
            for i in range(1, len(outcomes)):
                prev_outcome = outcomes[i-1]
                current_outcome = outcomes[i]
                
                # Calculate weight (more recent = higher weight)
                weight = self.discount_factor ** (len(outcomes) - i - 1)
                
                # Update distribution for the relevant side
                # Won if current outcome matches previous outcome
                if current_outcome == prev_outcome:
                    self.global_dist[prev_outcome]['alpha'] += weight
                else:
                    self.global_dist[prev_outcome]['beta'] += weight
        else:
            # Simple counting without weighting
            for i in range(1, len(outcomes)):
                prev_outcome = outcomes[i-1]
                current_outcome = outcomes[i]
                
                if current_outcome == prev_outcome:
                    self.global_dist[prev_outcome]['alpha'] += 1
                else:
                    self.global_dist[prev_outcome]['beta'] += 1
        
        # Update context-specific distributions if using context
        if self.use_context:
            for i in range(self.context_length, len(outcomes)):
                context = tuple(outcomes[i-self.context_length:i])
                current_outcome = outcomes[i]
                
                # For each bet option, update distribution based on whether it won
                for bet in ['P', 'B']:
                    if current_outcome == bet:
                        # Success: bet matched the outcome
                        self.context_dists[context][bet]['alpha'] += 1
                    else:
                        # Failure: bet didn't match the outcome
                        self.context_dists[context][bet]['beta'] += 1
    
    def _sample_win_probability(self, context, side):
        """
        Sample the win probability for a betting side from its Beta distribution.
        
        Args:
            context: Current context tuple or None
            side: 'P' or 'B' side to sample for
            
        Returns:
            float: Sampled win probability
        """
        # If using context and context exists with enough observations
        if context and self.use_context and context in self.context_dists:
            context_dist = self.context_dists[context][side]
            total_observations = context_dist['alpha'] + context_dist['beta'] - 2 * self.prior_alpha
            
            if total_observations >= self.min_context_observations:
                # Enough context-specific data, use it
                context_sample = beta.rvs(context_dist['alpha'], context_dist['beta'])
                global_sample = beta.rvs(self.global_dist[side]['alpha'], self.global_dist[side]['beta'])
                
                # Blend context-specific and global samples
                # As we get more context observations, rely more on context-specific distribution
                context_weight = min(0.8, total_observations / 20)  # Cap at 0.8 to always keep some global influence
                return context_sample * context_weight + global_sample * (1 - context_weight)
        
        # Fall back to global distribution if no valid context
        return beta.rvs(self.global_dist[side]['alpha'], self.global_dist[side]['beta'])
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "global_distributions": {
                'P': dict(self.global_dist['P']),
                'B': dict(self.global_dist['B'])
            },
            "context_count": len(self.context_dists),
            "recent_samples": self.recent_samples[-10:] if self.recent_samples else []
        }