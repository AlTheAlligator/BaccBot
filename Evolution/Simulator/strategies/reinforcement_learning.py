import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ReinforcementLearningStrategy(BaseStrategy):
    """
    Reinforcement Learning strategy for baccarat betting using Q-learning.
    
    This strategy learns optimal betting decisions over time by modeling the baccarat
    game as a Markov Decision Process and using Q-learning to discover which action
    (betting on Player, Banker, or skipping) maximizes expected reward in different states.
    """
    
    def __init__(self, game_simulator, params=None):
        """
        Initialize the reinforcement learning strategy.
        
        Args:
            game_simulator: GameSimulator instance
            params: Dict with optional parameters:
                - learning_rate: How quickly to update Q-values (default: 0.1)
                - discount_factor: Weight given to future rewards (default: 0.95)
                - exploration_rate: Probability of exploring vs exploiting (default: 0.2)
                - exploration_decay: Rate at which exploration decreases (default: 0.995)
                - min_exploration_rate: Minimum exploration rate (default: 0.01)
                - pattern_length: Length of outcome history to consider as state (default: 4)
                - feature_type: Method for state representation ('pattern', 'frequency', or 'combined')
                - use_adaptive_learning: Whether to adjust learning parameters based on performance
                - skip_confidence_threshold: Threshold for confidence below which to skip bets (default: 0.3)
                - bet_size: Flat bet size (default: 1)
        """
        # Call parent constructor correctly with both parameters
        super().__init__(game_simulator, params)
        
        # Initialize default parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.pattern_length = 4
        self.feature_type = 'combined'
        self.use_adaptive_learning = False
        self.skip_confidence_threshold = 0.3
        
        # Override with provided parameters
        if params:
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.discount_factor = params.get('discount_factor', self.discount_factor)
            self.exploration_rate = params.get('exploration_rate', self.exploration_rate)
            self.exploration_decay = params.get('exploration_decay', self.exploration_decay)
            self.min_exploration_rate = params.get('min_exploration_rate', self.min_exploration_rate)
            self.pattern_length = params.get('pattern_length', self.pattern_length)
            self.feature_type = params.get('feature_type', self.feature_type)
            self.use_adaptive_learning = params.get('use_adaptive_learning', self.use_adaptive_learning)
            self.skip_confidence_threshold = params.get('skip_confidence_threshold', self.skip_confidence_threshold)
        
        # Initialize Q-table with default values
        self.q_table = {}
        
        # Action space: 0 = Player, 1 = Banker, 2 = Skip
        self.actions = ['P', 'B', 'SKIP']
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.reward_history = []
        self.last_state = None
        self.last_action = None
        
        # For logging purposes
        logger.info(f"Initialized Reinforcement Learning Strategy with: LR={self.learning_rate}, " +
                   f"discount={self.discount_factor}, exploration={self.exploration_rate}")
    
    def get_state_representation(self, outcomes):
        """
        Convert the outcome history into a state representation for Q-learning.
        
        Args:
            outcomes: List of outcome strings ('P', 'B', 'T')
            
        Returns:
            tuple: State representation as a tuple for Q-table lookup
        """
        if not outcomes:
            # With no outcomes, return a default state
            return ('START',)
        
        # Only consider the most recent outcomes up to pattern_length
        recent = [o for o in outcomes[-self.pattern_length:] if o in ['P', 'B']]
        
        if self.feature_type == 'pattern':
            # Use the exact sequence of outcomes
            state = tuple(recent[-self.pattern_length:]) if recent else ('START',)
        
        elif self.feature_type == 'frequency':
            # Count frequencies of P and B
            p_count = recent.count('P')
            b_count = recent.count('B')
            state = (p_count, b_count) if recent else ('START',)
        
        else:  # 'combined' or default
            # Use both pattern and streaks
            if len(recent) >= self.pattern_length:
                # Get the pattern
                pattern = tuple(recent[-self.pattern_length:])
                
                # Calculate streak information
                if len(recent) >= 2:
                    streak = 1
                    current = recent[-1]
                    for i in range(len(recent)-2, -1, -1):
                        if recent[i] == current:
                            streak += 1
                        else:
                            break
                    streak_type = f"{current}:{streak}"
                else:
                    streak_type = "NONE"
                
                state = pattern + (streak_type,)
            else:
                # Not enough history
                state = ('INSUFFICIENT',)
        
        return state

    def get_q_value(self, state, action_idx):
        """
        Get the Q-value for a state-action pair, initializing if necessary.
        
        Args:
            state: The state representation (tuple)
            action_idx: Index of the action (0=P, 1=B, 2=SKIP)
            
        Returns:
            float: Q-value for the state-action pair
        """
        if state not in self.q_table:
            # Initialize Q-values for this state with small random values for exploration
            self.q_table[state] = np.random.uniform(0, 0.1, size=len(self.actions))
        
        return self.q_table[state][action_idx]

    def update_q_value(self, state, action_idx, reward, next_state):
        """
        Update the Q-value for a state-action pair using the Q-learning formula.
        
        Args:
            state: Current state
            action_idx: Index of the action taken
            reward: Reward received
            next_state: Resulting state
            
        Returns:
            float: Updated Q-value
        """
        # Get current Q value
        current_q = self.get_q_value(state, action_idx)
        
        # Get max Q value for next state
        if next_state not in self.q_table:
            # Initialize if this is a new state
            self.q_table[next_state] = np.random.uniform(0, 0.1, size=len(self.actions))
        
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action_idx] = new_q
        
        return new_q

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: The current state
            
        Returns:
            int: Index of the chosen action
        """
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            action_idx = np.random.choice(len(self.actions))
        else:
            # Exploitation: choose the best known action
            action_values = self.q_table.get(state, np.zeros(len(self.actions)))
            
            # Check for minimum confidence to avoid low-confidence bets
            max_q = np.max(action_values)
            skip_q = action_values[2]  # Q-value for skipping
            
            # If the highest Q-value doesn't exceed the skip confidence threshold
            # and skipping is an option, then skip
            if max_q < self.skip_confidence_threshold and skip_q >= max_q * 0.8:
                action_idx = 2  # Skip index
            else:
                # Get the action with highest Q-value, breaking ties randomly
                best_actions = np.where(action_values == max_q)[0]
                action_idx = np.random.choice(best_actions)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        return action_idx

    def receive_reward(self, outcome, bet):
        """
        Calculate the reward from an outcome and update Q-values.
        
        Args:
            outcome: The actual outcome ('P', 'B', or 'T')
            bet: The bet that was placed ('P', 'B', or 'SKIP')
            
        Returns:
            float: The calculated reward
        """
        if bet == 'SKIP':
            # Small negative reward for skipping, to encourage betting when confident
            reward = -0.05
        elif outcome == 'T':
            # Tie - no reward or penalty
            reward = 0
        elif outcome == bet:
            # Win
            reward = 0.95 if bet == 'B' else 1.0  # Account for banker commission
            self.wins += 1
        else:
            # Loss
            reward = -1.0
            self.losses += 1
        
        self.reward_history.append(reward)
        
        # If we have last state/action, update Q-value
        if self.last_state and self.last_action is not None:
            # Get action index
            action_idx = self.actions.index(self.last_action)
            
            # Update Q-table with this experience
            self.update_q_value(self.last_state, action_idx, reward, self.current_state)
        
        return reward

    def get_bet(self, outcomes):
        """
        Get the next bet based on RL strategy.
        
        Args:
            outcomes: List of past outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' for Player, 'B' for Banker, or 'SKIP' to skip betting
        """
        # Validate outcomes as required by BaseStrategy
        self._validate_outcome_list(outcomes)
        
        # Get state representation
        self.current_state = self.get_state_representation(outcomes)
        
        # Choose action based on current state
        action_idx = self.choose_action(self.current_state)
        bet = self.actions[action_idx]
        
        # Store state/action for later update
        self.last_state = self.current_state
        self.last_action = bet
        
        # If using adaptive learning, adjust parameters based on performance
        if self.use_adaptive_learning and len(self.reward_history) >= 10:
            self.adapt_learning_parameters()
            
        # Log the decision for debugging
        if len(outcomes) > 0:
            logger.debug(f"RL decision: State={self.current_state}, Action={bet}, " +
                        f"Exploration rate={self.exploration_rate:.2f}")
        
        return bet
    
    def adapt_learning_parameters(self):
        """Adjust learning parameters based on recent performance"""
        # Calculate recent performance (last 10 bets)
        recent_rewards = self.reward_history[-10:]
        avg_reward = np.mean(recent_rewards)
        
        # Adjust learning rate based on performance volatility
        reward_std = np.std(recent_rewards)
        if reward_std > 0.5:  # High volatility
            self.learning_rate = min(0.3, self.learning_rate * 1.05)  # Increase learning rate
        else:
            self.learning_rate = max(0.01, self.learning_rate * 0.99)  # Decrease learning rate
            
        # Adjust exploration rate based on average reward
        if avg_reward < -0.3:  # Poor performance
            # Increase exploration to try new strategies
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        elif avg_reward > 0.3:  # Good performance
            # Decrease exploration to exploit good strategy
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * 0.95)
            
        # Log adjustments at debug level
        logger.debug(f"Adapted RL params - LR: {self.learning_rate:.3f}, Explore: {self.exploration_rate:.3f}")