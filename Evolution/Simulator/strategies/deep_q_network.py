"""
Deep Q-Network strategy implementation.
"""

import logging
import numpy as np
from collections import deque
import random
from .base_strategy import BaseStrategy

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeepQNetworkStrategy(BaseStrategy):
    """
    Deep Q-Network (DQN) strategy for baccarat betting.

    This strategy uses a deep neural network to approximate the Q-function in reinforcement
    learning, allowing it to learn complex patterns and make betting decisions based on
    expected future rewards.
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Deep Q-Network strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)

        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback mode for DQN strategy.")
            self.tf_available = False
        else:
            self.tf_available = True

        # DQN parameters
        self.learning_rate = params.get('learning_rate', 0.001)
        self.discount_factor = params.get('discount_factor', 0.95)
        self.exploration_rate = params.get('exploration_rate', 0.1)
        self.batch_size = params.get('batch_size', 32)
        self.memory_size = params.get('memory_size', 1000)

        # State representation parameters
        self.state_size = 10  # Number of features in state representation
        self.action_size = 2  # P, B (no SKIP)

        # Banker bias (slight edge due to lower commission)
        self.banker_bias = params.get('banker_bias', 0.01)

        # Initialize memory for experience replay
        self.memory = deque(maxlen=self.memory_size)

        # Initialize model
        if self.tf_available:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

        # For tracking state
        self.last_state = None
        self.last_action = None
        self.training_count = 0

        # Performance tracking
        self.wins = 0
        self.losses = 0

        logger.info(f"Initialized Deep Q-Network strategy with exploration_rate={self.exploration_rate}")

    def _build_model(self):
        """
        Build a neural network model for DQN.

        Returns:
            Keras model
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update the target model with weights from the primary model."""
        if self.tf_available:
            self.target_model.set_weights(self.model.get_weights())

    def get_state(self, outcomes):
        """
        Convert outcomes to a state representation for the DQN.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            numpy array: State representation
        """
        # Filter out ties
        filtered = [o for o in outcomes if o in ['P', 'B']]

        if len(filtered) < 10:
            # Pad with balanced outcomes if not enough history
            padding = ['P', 'B'] * 5
            filtered = padding + filtered
            filtered = filtered[-10:]
        else:
            filtered = filtered[-10:]

        # Convert to numeric representation
        state = np.zeros(self.state_size)
        for i, outcome in enumerate(filtered):
            state[i] = 1 if outcome == 'P' else 0

        return state

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory for replay.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether this is a terminal state
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            int: Index of chosen action (0 for P, 1 for B)
        """
        if not self.tf_available or np.random.rand() <= self.exploration_rate:
            # Exploration: choose random action
            return random.randrange(self.action_size)

        # Exploitation: choose best action from Q values
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)

        # Apply banker bias to favor banker slightly (index 1)
        act_values[0][1] += self.banker_bias

        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Train the model using experience replay.

        Args:
            batch_size: Number of samples to use for training
        """
        if not self.tf_available or len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )

            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target

            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        self.training_count += 1

        # Periodically update target network
        if self.training_count % 10 == 0:
            self.update_target_model()

    def get_bet(self, outcomes):
        """
        Determine the next bet using the DQN.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)

        # Get current state
        state = self.get_state(outcomes)

        # Choose action
        action_idx = self.act(state)
        actions = ['P', 'B']
        bet = actions[action_idx]

        # Store state and action for later update
        self.last_state = state
        self.last_action = action_idx

        logger.debug(f"DQN decision: Action={bet}, Exploration rate={self.exploration_rate}")

        return bet

    def update_model(self, outcome, current_outcomes):
        """
        Update the model based on the outcome of the last bet.

        Args:
            outcome: The actual outcome ('P', 'B', or 'T')
            current_outcomes: The current list of outcomes including the new outcome
        """
        if self.last_state is None or self.last_action is None:
            return

        # Calculate reward
        actions = ['P', 'B']
        bet = actions[self.last_action]

        if outcome == 'T':
            reward = 0  # Tie
        elif outcome == bet:
            reward = 0.95 if bet == 'B' else 1.0  # Win (with banker commission)
            self.wins += 1
        else:
            reward = -1.0  # Loss
            self.losses += 1

        # Get next state
        next_state = self.get_state(current_outcomes)

        # Store experience
        self.remember(self.last_state, self.last_action, reward, next_state, False)

        # Train model
        if len(self.memory) >= self.batch_size:
            self.replay(self.batch_size)

        # Reset for next round
        self.last_state = None
        self.last_action = None

    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0,
            "memory_size": len(self.memory),
            "training_count": self.training_count
        }
