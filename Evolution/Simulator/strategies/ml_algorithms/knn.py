"""
K-Nearest Neighbors Algorithm for Baccarat Prediction

This module implements a K-Nearest Neighbors algorithm for predicting baccarat outcomes.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple

class KNNPredictor:
    """
    K-Nearest Neighbors predictor for baccarat outcomes.
    
    Features:
    - Distance-weighted voting
    - Confidence estimation
    - Incremental updates
    """
    
    def __init__(self, n_neighbors=5, weights='distance'):
        """
        Initialize the KNN predictor.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weighting scheme ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X = None  # Feature matrix
        self.y = None  # Labels
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        self.X = X
        self.y = y
        
    def update(self, x: np.ndarray, y: int):
        """
        Update the model with a new sample.
        
        Args:
            x: Feature vector
            y: Label
        """
        if self.X is None:
            self.X = x.reshape(1, -1)
            self.y = np.array([y])
        else:
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)
            
    def _get_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate distances from a point to all training samples.
        
        Args:
            x: Feature vector
            
        Returns:
            np.ndarray: Distances
        """
        if self.X is None:
            return np.array([])
            
        # Calculate Euclidean distances
        return np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        
    def predict(self, x: np.ndarray) -> int:
        """
        Predict the class for a sample.
        
        Args:
            x: Feature vector
            
        Returns:
            int: Predicted class
        """
        if self.X is None or len(self.X) < self.n_neighbors:
            return 0  # Default to Banker if not enough training data
            
        # Calculate distances
        distances = self._get_distances(x)
        
        # Get indices of k nearest neighbors
        k = min(self.n_neighbors, len(distances))
        nearest_indices = np.argsort(distances)[:k]
        
        # Get labels of nearest neighbors
        nearest_labels = self.y[nearest_indices]
        
        # Weighted voting
        if self.weights == 'distance':
            # Get weights (inverse of distance)
            nearest_distances = distances[nearest_indices]
            # Add small constant to avoid division by zero
            weights = 1.0 / (nearest_distances + 1e-6)
            
            # Count weighted votes
            player_votes = sum(weights[i] for i in range(k) if nearest_labels[i] == 1)
            banker_votes = sum(weights[i] for i in range(k) if nearest_labels[i] == 0)
            
            return 1 if player_votes > banker_votes else 0
        else:
            # Uniform weights (simple majority)
            votes = Counter(nearest_labels)
            return votes.most_common(1)[0][0]
            
    def predict_proba(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Predict class probabilities.
        
        Args:
            x: Feature vector
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if self.X is None or len(self.X) < self.n_neighbors:
            return 0, 0.5  # Default to Banker with neutral confidence
            
        # Calculate distances
        distances = self._get_distances(x)
        
        # Get indices of k nearest neighbors
        k = min(self.n_neighbors, len(distances))
        nearest_indices = np.argsort(distances)[:k]
        
        # Get labels of nearest neighbors
        nearest_labels = self.y[nearest_indices]
        
        # Weighted voting
        if self.weights == 'distance':
            # Get weights (inverse of distance)
            nearest_distances = distances[nearest_indices]
            # Add small constant to avoid division by zero
            weights = 1.0 / (nearest_distances + 1e-6)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Count weighted votes
            player_votes = sum(weights[i] for i in range(k) if nearest_labels[i] == 1)
            banker_votes = sum(weights[i] for i in range(k) if nearest_labels[i] == 0)
            
            total_votes = player_votes + banker_votes
            
            if player_votes > banker_votes:
                return 1, player_votes / total_votes
            else:
                return 0, banker_votes / total_votes
        else:
            # Uniform weights (simple majority)
            votes = Counter(nearest_labels)
            top_class, top_count = votes.most_common(1)[0]
            confidence = top_count / k
            
            return top_class, confidence
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            "algorithm": "KNN",
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "training_samples": len(self.X) if self.X is not None else 0
        }
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (not directly available for KNN).
        
        Returns:
            np.ndarray: Feature importance (uniform)
        """
        if self.X is None:
            return np.array([])
            
        # KNN doesn't have direct feature importance, return uniform importance
        return np.ones(self.X.shape[1]) / self.X.shape[1]
