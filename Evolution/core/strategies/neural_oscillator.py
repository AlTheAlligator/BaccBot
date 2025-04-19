"""
Neural Oscillator strategy implementation.
"""

import logging
import numpy as np
import math
from collections import deque

logger = logging.getLogger(__name__)

class NeuralOscillatorStrategy():
    """
    Neural Oscillator strategy that models the game as a system of coupled oscillators.

    This strategy is inspired by neural oscillation patterns in the brain and uses
    concepts like phase synchronization, resonance, and frequency adaptation to
    predict outcomes in a seemingly random sequence.
    """

    def __init__(self, params=None):
        """
        Initialize the Neural Oscillator strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """

        params = params or {}
        # Strategy parameters
        self.num_oscillators = params.get('num_oscillators', 8)
        self.coupling_strength = params.get('coupling_strength', 0.863673469387755)
        self.adaptation_rate = params.get('adaptation_rate', 0.3230612244897959)
        self.resonance_threshold = params.get('resonance_threshold', 0.8510204081632654)
        self.phase_sync_threshold = params.get('phase_sync_threshold', 0.1979591836734694)
        self.min_samples = params.get('min_samples', 6)
        self.banker_bias = params.get('banker_bias', 0.18947368421052632)
        self.memory_length = params.get('memory_length', 100)

        # Initialize oscillators
        self.frequencies = np.linspace(0.1, 0.5, self.num_oscillators)  # Base frequencies

        # Handle phases - either use provided phases or generate random ones
        if 'phases' in params and params['phases'] is not None:
            # Use provided phases
            phases_param = params['phases']

            # If phases is a list/array of the right length, use it directly
            if isinstance(phases_param, (list, np.ndarray)) and len(phases_param) == self.num_oscillators:
                self.phases = np.array(phases_param)
            # If it's a single value, create an array with evenly spaced phases starting from that value
            elif isinstance(phases_param, (int, float)):
                self.phases = np.linspace(phases_param, phases_param + 2*np.pi * (1 - 1/self.num_oscillators), self.num_oscillators) % (2*np.pi)
            else:
                # Default to random phases
                self.phases = np.random.uniform(0, 2*np.pi, self.num_oscillators)
                logger.warning(f"Invalid phases parameter: {phases_param}. Using random phases.")
        else:
            # Default to random phases
            self.phases = np.random.uniform(0, 2*np.pi, self.num_oscillators)

        self.phases = np.array([5.802010887463774, 6.244493748398569, 6.225044718659839, 6.227793810628512, 4.195115270151894, 6.203182816468961, 1.978748401676861, 3.581867800452434])

        # Ensure phases are within [0, 2π]
        self.phases = self.phases % (2*np.pi)
        self.amplitudes = np.ones(self.num_oscillators)  # Initial amplitudes

        # Initialize memory
        self.outcome_memory = deque(maxlen=self.memory_length)
        self.oscillator_performance = np.ones(self.num_oscillators) / self.num_oscillators

        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0

    def _update_oscillators(self, outcomes):
        """
        Update oscillator phases and frequencies based on recent outcomes.

        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Convert outcomes to numeric (0 for B, 1 for P)
        numeric = [1 if o == 'P' else 0 for o in outcomes[-self.num_oscillators:]]

        # Update phases based on outcomes and coupling
        for i in range(self.num_oscillators):
            # Natural frequency component
            self.phases[i] += 2 * np.pi * self.frequencies[i]

            # Coupling component
            for j in range(self.num_oscillators):
                if i != j:
                    phase_diff = self.phases[j] - self.phases[i]
                    self.phases[i] += self.coupling_strength * np.sin(phase_diff)

            # Outcome influence
            if i < len(numeric):
                target_phase = numeric[i] * np.pi  # 0 or π based on outcome
                phase_diff = target_phase - self.phases[i]
                self.phases[i] += self.adaptation_rate * np.sin(phase_diff)

            # Keep phase in [0, 2π]
            self.phases[i] = self.phases[i] % (2 * np.pi)

        # Update frequencies based on performance
        if len(self.outcome_memory) > 0:
            for i in range(self.num_oscillators):
                # Adjust frequency based on performance
                self.frequencies[i] *= (1 + self.adaptation_rate * (self.oscillator_performance[i] - 0.5))
                # Keep frequencies in reasonable range
                self.frequencies[i] = max(0.05, min(0.95, self.frequencies[i]))

    def _calculate_phase_synchronization(self):
        """
        Calculate phase synchronization index between oscillators.

        Returns:
            float: Synchronization index between 0 and 1
        """
        sync_index = 0
        count = 0

        for i in range(self.num_oscillators):
            for j in range(i+1, self.num_oscillators):
                # Calculate phase difference
                phase_diff = (self.phases[i] - self.phases[j]) % (2 * np.pi)
                # Convert to synchronization measure (1 when perfectly in sync)
                sync = 1 - (phase_diff / np.pi if phase_diff <= np.pi else (2*np.pi - phase_diff) / np.pi)
                sync_index += sync
                count += 1

        return sync_index / count if count > 0 else 0

    def _detect_resonance(self):
        """
        Detect if the oscillator system is in resonance.

        Returns:
            tuple: (is_in_resonance, resonance_strength)
        """
        # Calculate oscillator outputs
        outputs = np.sin(self.phases) * self.amplitudes

        # Calculate total system energy
        energy = np.sum(outputs**2)

        # Calculate coherence (how aligned the oscillators are)
        mean_output = np.mean(outputs)
        variance = np.mean((outputs - mean_output)**2)
        coherence = 1 - (variance / (np.mean(self.amplitudes**2) + 1e-10))

        # Resonance is high energy with high coherence
        resonance_strength = energy * coherence / self.num_oscillators
        is_in_resonance = resonance_strength > self.resonance_threshold

        return is_in_resonance, resonance_strength

    def _predict_next_outcome(self):
        """
        Predict the next outcome based on oscillator states.

        Returns:
            str: Predicted outcome ('P' or 'B')
        """
        # Calculate phase synchronization
        sync_index = self._calculate_phase_synchronization()

        # Detect resonance
        is_in_resonance, resonance_strength = self._detect_resonance()

        # Calculate oscillator outputs
        outputs = np.sin(self.phases) * self.amplitudes

        # Weight outputs by oscillator performance
        weighted_outputs = outputs * self.oscillator_performance

        # Calculate prediction score (positive favors P, negative favors B)
        prediction_score = np.sum(weighted_outputs)

        # Apply synchronization and resonance effects
        if is_in_resonance:
            # When in resonance, amplify the signal
            prediction_score *= (1 + resonance_strength)

        if sync_index > self.phase_sync_threshold:
            # When synchronized, make prediction more decisive
            prediction_score *= (1 + sync_index)

        # Apply banker bias
        prediction_score -= self.banker_bias

        logger.debug(f"Prediction score: {prediction_score:.3f}, Sync: {sync_index:.3f}, Resonance: {resonance_strength:.3f}")

        # Return prediction
        return 'P' if prediction_score > 0 else 'B'

    def _update_performance(self, prediction, actual):
        """
        Update performance metrics for oscillators.

        Args:
            prediction: Predicted outcome ('P' or 'B')
            actual: Actual outcome ('P' or 'B')
        """
        # Update overall performance
        correct = prediction == actual
        if correct:
            self.correct_predictions += 1
        self.total_predictions += 1

        # Update oscillator performance
        outputs = np.sin(self.phases) * self.amplitudes
        for i in range(self.num_oscillators):
            # Determine if this oscillator's output aligned with the correct prediction
            oscillator_prediction = 'P' if outputs[i] > 0 else 'B'
            oscillator_correct = oscillator_prediction == actual

            # Update performance (exponential moving average)
            self.oscillator_performance[i] = (
                0.9 * self.oscillator_performance[i] +
                0.1 * (1.0 if oscillator_correct else 0.0)
            )

    def get_bet(self, outcomes):
        """
        Determine the next bet using neural oscillator dynamics.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """

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

        # Update oscillators based on recent outcomes
        self._update_oscillators(filtered)

        # Make prediction
        prediction = self._predict_next_outcome()

        # Update performance if we have the actual outcome
        if len(self.outcome_memory) >= 2:
            self._update_performance(self.outcome_memory[-2], self.outcome_memory[-1])

        # Log performance
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logger.debug(f"Current accuracy: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")

        return prediction
