"""
Machine Learning Strategy

This strategy uses various machine learning algorithms to predict baccarat outcomes
based on historical patterns and engineered features.
"""

import numpy as np
from collections import deque
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
import random

# Import ML algorithm implementations
from .ml_algorithms.knn import KNNPredictor
from .ml_algorithms.linear_model import LinearModelPredictor
from .ml_algorithms.decision_tree import DecisionTreePredictor
from .ml_algorithms.ensemble import EnsemblePredictor

# Import feature engineering
from .ml_algorithms.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class MlStrategyStrategy:
    """
    A strategy that uses machine learning algorithms to predict baccarat outcomes.

    Features:
    - Multiple ML algorithm support (KNN, Linear Models, Decision Trees, Ensembles)
    - Automatic feature engineering
    - Online learning with incremental updates
    - Confidence-based betting
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Machine Learning strategy.

        Args:
            simulator: The simulator instance
            params: Dictionary of parameters for the strategy
        """
        self.simulator = simulator
        params = params or {}

        # Set fixed random seed for reproducibility
        # Use a seed derived from the strategy parameters for consistent but varied behavior
        seed = 42  # Default seed

        # If parameters are provided, create a seed from them
        if params:
            # Create a deterministic seed based on parameter values
            param_str = str(sorted([(k, str(v)) for k, v in params.items()]))
            seed = sum(ord(c) for c in param_str) % 10000

        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed  # Store for debugging

        # Core parameters
        self.algorithm = params.get('algorithm', 'ensemble')
        self.window_size = params.get('window_size', 10)
        self.min_samples = params.get('min_samples', 5)
        self.confidence_threshold = params.get('confidence_threshold', 0.55)
        self.banker_bias = params.get('banker_bias', 0.01)

        # Feature engineering parameters
        self.use_pattern_features = params.get('use_pattern_features', True)
        self.use_frequency_features = params.get('use_frequency_features', True)
        self.use_streak_features = params.get('use_streak_features', True)
        self.use_statistical_features = params.get('use_statistical_features', True)
        self.pattern_length = params.get('pattern_length', 3)

        # Algorithm-specific parameters
        self.knn_neighbors = params.get('knn_neighbors', 5)
        self.linear_model_type = params.get('linear_model_type', 'logistic')
        self.tree_max_depth = params.get('tree_max_depth', 3)

        # Handle ensemble_models parameter correctly
        if isinstance(params.get('ensemble_models'), list):
            self.ensemble_models = params.get('ensemble_models')
        else:
            # Default ensemble models
            self.ensemble_models = ['knn', 'linear', 'tree']

        # Learning parameters
        self.learning_rate = params.get('learning_rate', 0.1)
        self.regularization = params.get('regularization', 0.01)
        self.use_online_learning = params.get('use_online_learning', True)
        self.batch_update_size = params.get('batch_update_size', 10)

        # Add exploration parameter
        self.exploration_rate = params.get('exploration_rate', 0.1)  # 10% random exploration

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            use_pattern_features=self.use_pattern_features,
            use_frequency_features=self.use_frequency_features,
            use_streak_features=self.use_streak_features,
            use_statistical_features=self.use_statistical_features,
            pattern_length=self.pattern_length,
            window_size=self.window_size
        )

        # Initialize ML algorithm
        self._init_algorithm()

        # State tracking
        self.outcome_history = []
        self.feature_history = []
        self.label_history = []
        self.prediction_history = []
        self.confidence_history = []
        self.update_counter = 0

        # Debug info
        self.debug_info = {
            "random_decisions": 0,
            "model_decisions": 0,
            "banker_defaults": 0
        }

    def _init_algorithm(self):
        """Initialize the selected ML algorithm."""
        if self.algorithm == 'knn':
            self.predictor = KNNPredictor(
                n_neighbors=self.knn_neighbors
            )
        elif self.algorithm == 'linear':
            self.predictor = LinearModelPredictor(
                model_type=self.linear_model_type,
                learning_rate=self.learning_rate,
                regularization=self.regularization
            )
        elif self.algorithm == 'tree':
            self.predictor = DecisionTreePredictor(
                max_depth=self.tree_max_depth
            )
        elif self.algorithm == 'ensemble':
            # Create sub-predictors
            predictors = []
            if 'knn' in self.ensemble_models:
                predictors.append(KNNPredictor(n_neighbors=self.knn_neighbors))
            if 'linear' in self.ensemble_models:
                predictors.append(LinearModelPredictor(
                    model_type=self.linear_model_type,
                    learning_rate=self.learning_rate,
                    regularization=self.regularization
                ))
            if 'tree' in self.ensemble_models:
                predictors.append(DecisionTreePredictor(max_depth=self.tree_max_depth))

            self.predictor = EnsemblePredictor(predictors=predictors)
        else:
            # Default to ensemble if invalid algorithm specified
            logger.warning(f"Invalid algorithm '{self.algorithm}'. Using ensemble.")
            self.predictor = EnsemblePredictor(predictors=[
                KNNPredictor(n_neighbors=self.knn_neighbors),
                LinearModelPredictor(
                    model_type=self.linear_model_type,
                    learning_rate=self.learning_rate,
                    regularization=self.regularization
                ),
                DecisionTreePredictor(max_depth=self.tree_max_depth)
            ])

    def _prepare_features(self, outcomes: List[str]) -> np.ndarray:
        """
        Prepare features for the ML algorithm.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            np.ndarray: Feature vector
        """
        return self.feature_engineer.extract_features(outcomes)

    def _update_model(self, features: np.ndarray, label: int):
        """
        Update the ML model with new data.

        Args:
            features: Feature vector
            label: Outcome label (1 for Player, 0 for Banker)
        """
        # Store data for batch updates
        self.feature_history.append(features)
        self.label_history.append(label)

        # Perform online learning if enabled
        if self.use_online_learning:
            self.predictor.update(features, label)

        # Perform batch update if enough samples
        self.update_counter += 1
        if self.update_counter >= self.batch_update_size:
            # Convert to numpy arrays
            X = np.array(self.feature_history)
            y = np.array(self.label_history)

            # Fit model
            self.predictor.fit(X, y)

            # Reset counter
            self.update_counter = 0

    def get_bet(self, outcomes: List[str]) -> str:
        """
        Determine the next bet using machine learning.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        try:
            # Deterministic exploration based on game number
            # This ensures reproducibility while still allowing exploration
            if len(outcomes) > 0:
                # Use a hash of the outcome history length and seed for deterministic exploration
                exploration_hash = (len(outcomes) * self.seed) % 100
                if exploration_hash < self.exploration_rate * 100:
                    # Deterministic choice based on the hash
                    bet = 'P' if exploration_hash % 2 == 0 else 'B'
                    self.debug_info["random_decisions"] += 1
                    return bet

            # Always start from game 7
            if len(outcomes) < 7:
                self.debug_info["banker_defaults"] += 1
                return 'B'  # Default to Banker for initial games

            # Store outcome history
            self.outcome_history = outcomes.copy()

            # Not enough data for ML prediction
            if len(outcomes) < self.min_samples + self.pattern_length:
                self.debug_info["banker_defaults"] += 1
                return 'B'  # Default to Banker

            # Prepare features
            features = self._prepare_features(outcomes)

            # Make prediction if we have enough training data
            if len(self.feature_history) >= self.min_samples:
                # Get prediction and confidence
                prediction, confidence = self.predictor.predict_proba(features)

                # Store prediction and confidence
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)

                # Apply banker bias
                if prediction == 0:  # Banker prediction
                    confidence += confidence * self.banker_bias

                # Make decision based on confidence
                if confidence > self.confidence_threshold:
                    self.debug_info["model_decisions"] += 1
                    return 'P' if prediction == 1 else 'B'
                else:
                    # Not confident enough, default to Banker
                    self.debug_info["banker_defaults"] += 1
                    return 'B'
            else:
                # Not enough training data, default to Banker
                self.debug_info["banker_defaults"] += 1
                return 'B'

        except Exception as e:
            logger.error(f"Error in ML Strategy get_bet: {e}")
            # In case of error, default to Banker
            self.debug_info["banker_defaults"] += 1
            return 'B'

    def update_result(self, bet: str, outcome: str, win: bool):
        """
        Update strategy with the result of the last bet.

        Args:
            bet: The bet that was placed ('P' or 'B')
            outcome: The actual outcome ('P', 'B', or 'T')
            win: Whether the bet won
        """
        try:
            # Skip ties
            if outcome == 'T':
                return

            # Convert outcome to label
            label = 1 if outcome == 'P' else 0

            # Get features for the previous state
            if len(self.outcome_history) > 0:
                # Remove the last outcome (which we just observed)
                prev_outcomes = self.outcome_history[:-1]

                # Only update if we have enough history
                if len(prev_outcomes) >= self.min_samples + self.pattern_length:
                    features = self._prepare_features(prev_outcomes)
                    self._update_model(features, label)

                    # Log some debug info
                    if len(self.prediction_history) > 0 and len(self.label_history) > 0:
                        last_prediction = self.prediction_history[-1]
                        last_label = self.label_history[-1]
                        correct = last_prediction == last_label
                        logger.debug(f"ML prediction: {last_prediction}, actual: {last_label}, correct: {correct}")

        except Exception as e:
            logger.error(f"Error in ML Strategy update_result: {e}, bet={bet}, outcome={outcome}, win={win}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for debugging.

        Returns:
            dict: Strategy statistics
        """
        try:
            # Calculate prediction accuracy
            correct_predictions = 0
            total_predictions = 0

            for i in range(min(len(self.prediction_history), len(self.label_history))):
                pred = self.prediction_history[i]
                label = self.label_history[i]

                if pred == label:
                    correct_predictions += 1

                total_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Calculate average confidence
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0

            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.predictor, 'get_feature_importance'):
                importance = self.predictor.get_feature_importance()
                feature_names = self.feature_engineer.get_feature_names()

                for i, name in enumerate(feature_names):
                    if i < len(importance):
                        feature_importance[name] = importance[i]

            # Calculate decision distribution
            total_decisions = sum(self.debug_info.values())
            decision_distribution = {}
            if total_decisions > 0:
                for key, value in self.debug_info.items():
                    decision_distribution[key] = f"{value / total_decisions * 100:.1f}%"

            return {
                "strategy": f"ML Strategy ({self.algorithm})",
                "accuracy": f"{accuracy:.2f}",
                "avg_confidence": f"{avg_confidence:.2f}",
                "training_samples": len(self.feature_history),
                "decision_counts": self.debug_info,
                "decision_distribution": decision_distribution,
                "feature_importance": feature_importance,
                "algorithm_params": self.predictor.get_params(),
                "exploration_rate": self.exploration_rate,
                "random_seed": self.seed
            }
        except Exception as e:
            logger.error(f"Error in ML Strategy get_stats: {e}")
            return {
                "strategy": f"ML Strategy ({self.algorithm})",
                "error": f"Error generating stats: {e}"
            }
