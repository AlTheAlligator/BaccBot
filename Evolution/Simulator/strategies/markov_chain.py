import logging
from collections import defaultdict, Counter

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MarkovChainStrategy(BaseStrategy):
    """
    A strategy that uses a Markov chain model to predict the next outcome.
    
    This strategy builds a transition probability matrix based on observed outcome
    sequences and uses it to predict the most likely next outcome.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Markov Chain strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.order = params.get('order', 2)  # Order of the Markov chain (sequence length)
        self.min_samples = params.get('min_samples', max(15, self.order * 4))  # Minimum samples before making predictions
        self.min_sequence_observations = params.get('min_sequence_observations', 3)  # Minimum times a sequence must be observed
        self.confidence_threshold = params.get('confidence_threshold', 0.6)  # Min probability to place bet
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.use_higher_order_fallback = params.get('use_higher_order_fallback', True)  # Whether to use higher-order chains as fallback
        self.use_smoothing = params.get('use_smoothing', True)  # Use Laplace smoothing for probabilities
        self.smoothing_factor = params.get('smoothing_factor', 0.5)  # Smoothing factor (pseudo-count)
        self.use_adaptive_order = params.get('use_adaptive_order', False)  # Whether to adapt order based on data
        self.max_order = params.get('max_order', 4)  # Maximum order when using adaptive order
        
        # Initialize transition matrices for different orders
        self.transitions = {}
        for i in range(1, self.max_order + 1):
            self.transitions[i] = defaultdict(Counter)
            
        # For tracking state
        self.bet_count = 0
        self.last_state = None
        
        logger.info(f"Initialized Markov Chain strategy with order={self.order}, confidence_threshold={self.confidence_threshold}")
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on Markov chain prediction.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data to build reliable transition matrix
            logger.debug(f"Not enough data for Markov chain ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Filter out ties for Markov chain analysis
        non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
        if len(non_tie_outcomes) < self.min_samples:
            logger.debug(f"Not enough non-tie outcomes ({len(non_tie_outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Update transition matrices with observed data
        self._update_transitions(non_tie_outcomes)
        
        # Determine effective order to use (adaptive if configured)
        effective_order = self._get_effective_order(non_tie_outcomes)
        
        # Get current state (sequence of last N outcomes)
        if len(non_tie_outcomes) >= effective_order:
            current_state = tuple(non_tie_outcomes[-effective_order:])
        else:
            # Not enough data for the current order
            logger.debug(f"Not enough data for order {effective_order}")
            return "SKIP"
        
        # Check if we've observed this state enough times
        state_count = sum(self.transitions[effective_order][current_state].values())
        if state_count < self.min_sequence_observations:
            # Try fallback to lower orders if configured
            if self.use_higher_order_fallback:
                return self._try_order_fallbacks(non_tie_outcomes)
            else:
                logger.debug(f"Not enough observations of state {current_state}: {state_count} < {self.min_sequence_observations}")
                return "SKIP"
        
        # Get transition probabilities for current state
        p_prob, b_prob = self._get_transition_probs(current_state, effective_order)
        
        # Apply banker bias
        b_prob += self.banker_bias
        
        # Normalize probabilities
        total = p_prob + b_prob
        p_prob /= total
        b_prob /= total
        
        logger.debug(f"Markov probabilities for state {current_state}: P={p_prob:.3f}, B={b_prob:.3f}")
        
        # Make decision based on transition probabilities
        if max(p_prob, b_prob) >= self.confidence_threshold:
            # Track this bet
            self.bet_count += 1
            self.last_state = current_state
            
            return "P" if p_prob > b_prob else "B"
        else:
            logger.debug(f"Confidence too low: max prob {max(p_prob, b_prob):.3f} < {self.confidence_threshold}")
            return "SKIP"
    
    def _update_transitions(self, outcomes):
        """
        Update transition matrices based on observed outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Update transitions for each order
        for order in range(1, self.max_order + 1):
            if len(outcomes) <= order:
                continue
                
            # Process each possible state and its transition
            for i in range(len(outcomes) - order):
                state = tuple(outcomes[i:i+order])
                next_outcome = outcomes[i+order]
                
                # Update transition count from this state to next outcome
                self.transitions[order][state][next_outcome] += 1
    
    def _get_effective_order(self, outcomes):
        """
        Determine the effective Markov chain order to use.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            int: Effective order to use
        """
        if not self.use_adaptive_order:
            return self.order
            
        # Adaptive order logic - use highest order with enough data
        for order in range(self.max_order, 0, -1):
            # Check if we have enough data for this order
            min_observations_for_order = self.min_sequence_observations * (2 ** order)  # Exponential growth of state space
            
            if len(outcomes) >= min_observations_for_order:
                return min(order, self.max_order)
                
        return 1  # Fallback to first-order
    
    def _get_transition_probs(self, state, order):
        """
        Get transition probabilities from a state.
        
        Args:
            state: Current state (tuple of outcomes)
            order: Order of the Markov chain
            
        Returns:
            tuple: (p_prob, b_prob) probabilities
        """
        transitions = self.transitions[order][state]
        
        if self.use_smoothing:
            # Apply Laplace smoothing
            p_count = transitions.get('P', 0) + self.smoothing_factor
            b_count = transitions.get('B', 0) + self.smoothing_factor
            total = p_count + b_count
        else:
            # Raw counts
            p_count = transitions.get('P', 0)
            b_count = transitions.get('B', 0)
            total = p_count + b_count
        
        if total > 0:
            p_prob = p_count / total
            b_prob = b_count / total
        else:
            p_prob = b_prob = 0.5
            
        return p_prob, b_prob
    
    def _try_order_fallbacks(self, outcomes):
        """
        Try fallbacks to lower-order chains when higher-order chain has insufficient data.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: 'P', 'B', or 'SKIP'
        """
        effective_order = self._get_effective_order(outcomes)
        
        # Try progressively lower orders
        for order in range(effective_order - 1, 0, -1):
            if len(outcomes) >= order:
                current_state = tuple(outcomes[-order:])
                state_count = sum(self.transitions[order][current_state].values())
                
                if state_count >= self.min_sequence_observations:
                    # Get transition probabilities for this order
                    p_prob, b_prob = self._get_transition_probs(current_state, order)
                    
                    # Apply banker bias
                    b_prob += self.banker_bias
                    
                    # Normalize
                    total = p_prob + b_prob
                    p_prob /= total
                    b_prob /= total
                    
                    # Reduce confidence threshold slightly for fallbacks
                    adjusted_threshold = self.confidence_threshold - (0.03 * (effective_order - order))
                    
                    logger.debug(f"Fallback to order {order}, state {current_state}: P={p_prob:.3f}, B={b_prob:.3f}, "
                                f"threshold={adjusted_threshold:.3f}")
                    
                    if max(p_prob, b_prob) >= adjusted_threshold:
                        self.bet_count += 1
                        self.last_state = current_state
                        
                        return "P" if p_prob > b_prob else "B"
        
        return "SKIP"
    
    def get_transition_matrices(self):
        """Get transition matrices for debugging."""
        matrix_stats = {}
        
        for order in range(1, self.max_order + 1):
            if order in self.transitions:
                # Convert to a serializable format
                matrix = {}
                
                for state, transitions in self.transitions[order].items():
                    # Only include states with enough observations
                    if sum(transitions.values()) >= self.min_sequence_observations:
                        matrix[str(state)] = dict(transitions)
                
                matrix_stats[str(order)] = {
                    "states_count": len(matrix),
                    "top_states": sorted(matrix.items(), 
                                        key=lambda x: sum(x[1].values()), 
                                        reverse=True)[:5] if matrix else []
                }
                
        return {
            "matrices": matrix_stats,
            "bets_made": self.bet_count,
            "last_state": str(self.last_state) if self.last_state else None
        }