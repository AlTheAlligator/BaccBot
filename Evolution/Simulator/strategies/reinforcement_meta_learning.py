"""
Reinforcement Meta-Learning strategy implementation.
"""

import logging
import numpy as np
import random
from collections import defaultdict, deque
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ReinforcementMetaLearningStrategy(BaseStrategy):
    """
    Reinforcement Meta-Learning strategy that learns how to learn from past betting experiences.
    
    This strategy implements a meta-learning algorithm that adapts its learning rate,
    uses a portfolio of sub-strategies that compete for influence, implements exploration
    vs. exploitation trade-offs, and uses contextual bandits to select the best strategy.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Reinforcement Meta-Learning strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.num_strategies = params.get('num_strategies', 5)
        self.initial_learning_rate = params.get('initial_learning_rate', 0.1)
        self.exploration_rate = params.get('exploration_rate', 0.2)
        self.exploration_decay = params.get('exploration_decay', 0.995)
        self.min_exploration = params.get('min_exploration', 0.05)
        self.reward_discount = params.get('reward_discount', 0.9)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.memory_length = params.get('memory_length', 200)
        self.context_length = params.get('context_length', 5)
        
        # Initialize sub-strategies
        self.strategies = [
            {'name': 'frequency', 'weight': 1.0, 'correct': 0, 'total': 0},
            {'name': 'streak', 'weight': 1.0, 'correct': 0, 'total': 0},
            {'name': 'pattern', 'weight': 1.0, 'correct': 0, 'total': 0},
            {'name': 'alternating', 'weight': 1.0, 'correct': 0, 'total': 0},
            {'name': 'adaptive', 'weight': 1.0, 'correct': 0, 'total': 0}
        ]
        
        # Initialize contextual bandit
        self.contexts = {}  # Maps context to strategy weights
        
        # Initialize meta-learning parameters
        self.learning_rate = self.initial_learning_rate
        self.meta_learning_rate = 0.01
        
        # Initialize memory
        self.outcome_memory = deque(maxlen=self.memory_length)
        self.prediction_memory = deque(maxlen=self.memory_length)
        self.strategy_predictions = deque(maxlen=self.memory_length)
        
        # Initialize pattern store for pattern matching
        self.pattern_store = defaultdict(lambda: {'P': 0, 'B': 0})
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
    
    def _get_context(self, outcomes):
        """
        Extract context from recent outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Context string
        """
        if len(outcomes) < self.context_length:
            return 'default'
        
        # Use last context_length outcomes as context
        context = ''.join(outcomes[-self.context_length:])
        return context
    
    def _get_strategy_weights(self, context):
        """
        Get strategy weights for a specific context.
        
        Args:
            context: Context string
            
        Returns:
            list: Strategy weights
        """
        if context not in self.contexts:
            # Initialize weights for new context
            self.contexts[context] = np.ones(len(self.strategies)) / len(self.strategies)
        
        return self.contexts[context]
    
    def _update_strategy_weights(self, context, strategy_idx, reward):
        """
        Update strategy weights for a specific context.
        
        Args:
            context: Context string
            strategy_idx: Index of strategy to update
            reward: Reward value (1 for correct, 0 for incorrect)
        """
        if context not in self.contexts:
            return
        
        # Update weight for the selected strategy
        weights = self.contexts[context]
        weights[strategy_idx] += self.learning_rate * reward
        
        # Normalize weights
        weights = np.maximum(0.01, weights)  # Ensure minimum weight
        self.contexts[context] = weights / np.sum(weights)
    
    def _update_meta_parameters(self, correct):
        """
        Update meta-learning parameters based on prediction accuracy.
        
        Args:
            correct: Whether the prediction was correct
        """
        if correct:
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
            
            # Decrease learning rate when consistently correct
            if self.consecutive_correct > 5:
                self.learning_rate = max(0.01, self.learning_rate * 0.95)
        else:
            self.consecutive_wrong += 1
            self.consecutive_correct = 0
            
            # Increase learning rate when consistently wrong
            if self.consecutive_wrong > 3:
                self.learning_rate = min(0.5, self.learning_rate * 1.1)
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
    
    def _frequency_strategy(self, outcomes, window_size=20):
        """
        Predict based on frequency analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            window_size: Size of window to analyze
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < window_size:
            window = outcomes
        else:
            window = outcomes[-window_size:]
        
        p_count = window.count('P')
        b_count = window.count('B')
        
        # Apply banker bias
        b_count += b_count * self.banker_bias
        
        return 'P' if p_count > b_count else 'B'
    
    def _streak_strategy(self, outcomes):
        """
        Predict based on streak analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < 3:
            return 'B'
        
        # Get current streak
        current = outcomes[-1]
        streak_length = 1
        for i in range(len(outcomes) - 2, -1, -1):
            if outcomes[i] == current:
                streak_length += 1
            else:
                break
        
        # If streak is long, predict it will break
        if streak_length >= 3:
            return 'P' if current == 'B' else 'B'
        # Otherwise predict it will continue
        else:
            return current
    
    def _pattern_strategy(self, outcomes, pattern_length=3):
        """
        Predict based on pattern analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            pattern_length: Length of pattern to analyze
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < pattern_length + 1:
            return 'B'
        
        # Get current pattern
        pattern = tuple(outcomes[-pattern_length:])
        
        # Check if pattern exists in store
        if pattern in self.pattern_store:
            counts = self.pattern_store[pattern]
            p_count = counts['P']
            b_count = counts['B']
            
            if p_count + b_count > 0:
                # Apply banker bias
                b_count += b_count * self.banker_bias
                
                # Normalize
                total = p_count + b_count
                p_prob = p_count / total
                b_prob = b_count / total
                
                return 'P' if p_prob > b_prob else 'B'
        
        return 'B'  # Default to Banker
    
    def _alternating_strategy(self, outcomes):
        """
        Predict based on alternating pattern analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < 4:
            return 'B'
        
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
    
    def _adaptive_strategy(self, outcomes):
        """
        Predict based on adaptive analysis.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        if len(outcomes) < 20:
            return 'B'
        
        # Calculate recent bias
        recent = outcomes[-20:]
        p_count = recent.count('P')
        b_count = recent.count('B')
        
        # Calculate trend
        first_half = outcomes[-20:-10]
        second_half = outcomes[-10:]
        
        p_first = first_half.count('P') / len(first_half)
        p_second = second_half.count('P') / len(second_half)
        
        p_trend = p_second - p_first
        
        # Combine bias and trend
        p_probability = (p_count / (p_count + b_count)) + 0.2 * p_trend
        
        # Apply banker bias
        p_probability -= self.banker_bias
        
        return 'P' if p_probability > 0.5 else 'B'
    
    def _update_pattern_store(self, outcomes):
        """
        Update pattern store with new outcomes.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        for pattern_length in range(2, 5):
            if len(outcomes) <= pattern_length:
                continue
                
            for i in range(len(outcomes) - pattern_length - 1):
                pattern = tuple(outcomes[i:i+pattern_length])
                next_outcome = outcomes[i+pattern_length]
                self.pattern_store[pattern][next_outcome] += 1
    
    def _select_strategy(self, context, strategy_predictions):
        """
        Select strategy using contextual bandit approach.
        
        Args:
            context: Context string
            strategy_predictions: List of predictions from each strategy
            
        Returns:
            tuple: (selected_strategy_idx, prediction)
        """
        weights = self._get_strategy_weights(context)
        
        # Exploration vs. exploitation
        if random.random() < self.exploration_rate:
            # Exploration - select random strategy
            selected_idx = random.choices(range(len(self.strategies)), 
                                         weights=weights, k=1)[0]
        else:
            # Exploitation - select best strategy
            selected_idx = np.argmax(weights)
        
        return selected_idx, strategy_predictions[selected_idx]
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using reinforcement meta-learning.
        
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
        
        # Update pattern store
        self._update_pattern_store(filtered)
        
        # Get predictions from each strategy
        strategy_predictions = [
            self._frequency_strategy(filtered),
            self._streak_strategy(filtered),
            self._pattern_strategy(filtered),
            self._alternating_strategy(filtered),
            self._adaptive_strategy(filtered)
        ]
        
        # Get current context
        context = self._get_context(filtered)
        
        # Select strategy
        selected_idx, prediction = self._select_strategy(context, strategy_predictions)
        
        # Store prediction and strategy predictions
        self.prediction_memory.append(prediction)
        self.strategy_predictions.append(strategy_predictions)
        
        # Update strategy performance and weights if we have the actual outcome
        if len(self.outcome_memory) >= 2 and len(self.prediction_memory) >= 2:
            actual = self.outcome_memory[-1]
            prev_prediction = self.prediction_memory[-2]
            prev_strategy_predictions = self.strategy_predictions[-2]
            prev_context = self._get_context(list(self.outcome_memory)[:-1])
            
            # Check if prediction was correct
            correct = prev_prediction == actual
            
            # Update overall performance
            if correct:
                self.correct_predictions += 1
            self.total_predictions += 1
            
            # Update strategy performance
            for i, strategy_pred in enumerate(prev_strategy_predictions):
                strategy_correct = strategy_pred == actual
                self.strategies[i]['total'] += 1
                if strategy_correct:
                    self.strategies[i]['correct'] += 1
                
                # Update strategy weight
                reward = 1 if strategy_correct else -0.5
                self._update_strategy_weights(prev_context, i, reward)
            
            # Update meta-parameters
            self._update_meta_parameters(correct)
        
        # Log performance and parameters
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logger.debug(f"Current accuracy: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")
            logger.debug(f"Learning rate: {self.learning_rate:.3f}, Exploration rate: {self.exploration_rate:.3f}")
            
            # Log strategy performance
            for strategy in self.strategies:
                if strategy['total'] > 0:
                    strategy_accuracy = strategy['correct'] / strategy['total']
                    logger.debug(f"{strategy['name']} accuracy: {strategy_accuracy:.3f}")
        
        return prediction
