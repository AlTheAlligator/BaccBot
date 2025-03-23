import logging
import numpy as np
from collections import defaultdict, deque
import importlib
import sys

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MetaStrategyResponse:
    """Helper class to track individual strategy responses and performance"""
    
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.decisions = []  # List of (bet, outcome) tuples
        self.performance = 0.0  # Cumulative performance score
        self.weight = 1.0  # Dynamic weight in ensemble
        self.recent_correct = deque(maxlen=10)  # Track recent accuracy
    
    def add_decision(self, bet, actual_outcome):
        """Record a strategy decision and update its performance metrics"""
        # Skip ties in evaluation
        if actual_outcome == 'T':
            return
            
        self.decisions.append((bet, actual_outcome))
        
        # Calculate result of this decision
        correct = (bet == actual_outcome)
        
        # Update recent accuracy tracking
        self.recent_correct.append(1 if correct else 0)
        
        # Update performance score
        # Win: +1, Loss: -1.05 (to account for banker commission)
        if correct:
            self.performance += 1.0
        else:
            self.performance -= 1.05
            
        # Update weight based on recent performance
        if self.recent_correct:
            recent_accuracy = sum(self.recent_correct) / len(self.recent_correct)
            # Adjust weight but keep it within reasonable bounds
            self.weight = max(0.1, min(2.0, 0.5 + recent_accuracy))

class MetaStrategy(BaseStrategy):
    """
    A meta-strategy that dynamically selects among other strategies based on their performance.
    
    This strategy manages and evaluates multiple sub-strategies, learning which strategies
    perform best in different contexts and dynamically weighting their decisions.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Meta Strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Default parameters
        self.min_samples = params.get('min_samples', 15)  # Minimum samples before making predictions
        self.confidence_threshold = params.get('confidence_threshold', 0.6)  # Min confidence to place bet
        self.banker_bias = params.get('banker_bias', 0.01)  # Slight bias towards banker bets
        self.use_weighted_voting = params.get('use_weighted_voting', True)  # Whether to weight strategies by performance
        self.adaptation_rate = params.get('adaptation_rate', 0.1)  # Rate at which strategy weights are updated
        self.exploration_factor = params.get('exploration_factor', 0.05)  # Probability to try underperforming strategies
        
        # List of strategies to use
        self.strategy_names = params.get('strategies', [
            'adaptive_bias',
            'pattern_based',
            'majority_last_n',
            'markov_chain',
            'thompson_sampling'
        ])
        
        # Dictionary to keep track of each strategy's performance
        self.strategy_trackers = {}
        
        # List to track meta-strategy decisions
        self.meta_decisions = []
        
        # Initialize the sub-strategies
        self._initialize_strategies()
        
        logger.info(f"Initialized Meta Strategy with {len(self.strategies)} sub-strategies: {', '.join(self.strategy_names)}")
    
    def _initialize_strategies(self):
        """Initialize all sub-strategy instances"""
        self.strategies = {}
        
        # Import the base_strategy module
        try:
            base_module = sys.modules['strategies.base_strategy']
            
            # Import and initialize each strategy
            for strategy_name in self.strategy_names:
                try:
                    # Import the strategy module
                    module_name = f"strategies.{strategy_name}"
                    module = importlib.import_module(module_name)
                    
                    # Construct class name from strategy name (e.g., adaptive_bias -> AdaptiveBiasStrategy)
                    class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
                    
                    # Get the strategy class
                    strategy_class = getattr(module, class_name)
                    
                    # Create instance with appropriate parameters
                    strategy_params = self.params.get(strategy_name, {})
                    strategy_instance = strategy_class(self.simulator, strategy_params)
                    
                    # Store the instance
                    self.strategies[strategy_name] = strategy_instance
                    
                    # Initialize tracker for this strategy
                    self.strategy_trackers[strategy_name] = MetaStrategyResponse(strategy_name)
                    
                    logger.info(f"Initialized sub-strategy: {strategy_name}")
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to initialize strategy '{strategy_name}': {e}")
        except Exception as e:
            logger.error(f"Error initializing sub-strategies: {e}")
            # Initialize with empty strategies dict as fallback
            self.strategies = {}
    
    def get_bet(self, outcomes):
        """
        Get the next bet based on the ensemble of sub-strategy recommendations.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P', 'B', or 'SKIP' for the next bet
        """
        if len(outcomes) < self.min_samples:
            # Not enough data for reliable meta-strategy decisions
            logger.debug(f"Not enough data for meta-strategy ({len(outcomes)} < {self.min_samples})")
            return "SKIP"
        
        # Collect recommendations from all sub-strategies
        recommendations = {}
        for name, strategy in self.strategies.items():
            try:
                bet = strategy.get_bet(outcomes)
                recommendations[name] = bet
                logger.debug(f"Strategy '{name}' recommends: {bet}")
            except Exception as e:
                logger.error(f"Error getting bet from strategy '{name}': {e}")
                recommendations[name] = "SKIP"
        
        # If previous outcome is available, update performance tracking
        if len(self.meta_decisions) > 0 and len(outcomes) > 0:
            last_decision = self.meta_decisions[-1]
            prev_outcome = outcomes[-1]
            
            # Update each strategy's performance
            for name, bet in last_decision.get('recommendations', {}).items():
                if name in self.strategy_trackers and bet != "SKIP":
                    tracker = self.strategy_trackers[name]
                    tracker.add_decision(bet, prev_outcome)
        
        # Count votes for each bet option with optional weighting
        votes = {'P': 0, 'B': 0, 'SKIP': 0}
        
        for name, bet in recommendations.items():
            if bet in votes:
                if self.use_weighted_voting and name in self.strategy_trackers:
                    # Add weighted vote
                    votes[bet] += self.strategy_trackers[name].weight
                else:
                    # Add simple vote
                    votes[bet] += 1
        
        # Apply banker bias
        votes['B'] += self.banker_bias
        
        logger.debug(f"Vote counts: P={votes['P']:.2f}, B={votes['B']:.2f}, SKIP={votes['SKIP']:.2f}")
        
        # Record this decision
        self.meta_decisions.append({
            'recommendations': recommendations,
            'votes': votes
        })
        
        # Make decision based on votes
        if votes['SKIP'] > votes['P'] and votes['SKIP'] > votes['B']:
            return "SKIP"
            
        # If there's a clear winner between P and B
        if votes['P'] > votes['B']:
            confidence = votes['P'] / (votes['P'] + votes['B'] + votes['SKIP'])
            if confidence >= self.confidence_threshold:
                return "P"
            else:
                logger.debug(f"Confidence too low for P: {confidence:.2f} < {self.confidence_threshold}")
                return "SKIP"
        else:
            confidence = votes['B'] / (votes['P'] + votes['B'] + votes['SKIP'])
            if confidence >= self.confidence_threshold:
                return "B"
            else:
                logger.debug(f"Confidence too low for B: {confidence:.2f} < {self.confidence_threshold}")
                return "SKIP"
    
    def get_best_strategies(self, top_n=3):
        """Get the top performing strategies for debugging."""
        if not self.strategy_trackers:
            return []
            
        # Sort strategies by performance
        sorted_strategies = sorted(
            self.strategy_trackers.values(),
            key=lambda x: x.performance,
            reverse=True
        )
        
        return [
            {
                'name': s.strategy_name,
                'performance': s.performance,
                'weight': s.weight,
                'recent_accuracy': sum(s.recent_correct) / len(s.recent_correct) if s.recent_correct else 0
            }
            for s in sorted_strategies[:top_n]
        ]