"""
Decision Tree Algorithm for Baccarat Prediction

This module implements a simple decision tree algorithm for predicting baccarat outcomes.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

class Node:
    """Decision tree node."""
    
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a decision tree node.
        
        Args:
            feature_idx: Index of the feature to split on
            threshold: Threshold value for the split
            left: Left child node
            right: Right child node
            value: Predicted value (for leaf nodes)
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTreePredictor:
    """
    Decision tree predictor for baccarat outcomes.
    
    Features:
    - Binary classification tree
    - Information gain splitting criterion
    - Pruning with max depth
    - Feature importance estimation
    """
    
    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Initialize the decision tree predictor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_importance = None
        self.n_features = None
        
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a set of labels.
        
        Args:
            y: Labels
            
        Returns:
            float: Entropy value
        """
        # Count class frequencies
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        
        return entropy
        
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate information gain from a split.
        
        Args:
            y: Parent labels
            y_left: Left child labels
            y_right: Right child labels
            
        Returns:
            float: Information gain
        """
        # Calculate parent entropy
        parent_entropy = self._entropy(y)
        
        # Calculate weighted average of children entropy
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n == 0 or n_left == 0 or n_right == 0:
            return 0
            
        child_entropy = (n_left / n) * self._entropy(y_left) + (n_right / n) * self._entropy(y_right)
        
        # Information gain is the difference
        return parent_entropy - child_entropy
        
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            tuple: (feature_idx, threshold) or (None, None) if no good split found
        """
        n_samples, n_features = X.shape
        
        # Initialize variables
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Try all features
        for feature_idx in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try all thresholds
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Skip if split is degenerate
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                # Calculate information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Feature matrix
            y: Labels
            depth: Current depth
            
        Returns:
            Node: Root node of the tree
        """
        n_samples, n_features = X.shape
        
        # Update feature importance tracking
        if self.feature_importance is None:
            self.feature_importance = np.zeros(n_features)
            self.n_features = n_features
            
        # Check stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            # Create leaf node
            counter = Counter(y)
            value = counter.most_common(1)[0][0]
            return Node(value=value)
            
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        # If no good split found, create leaf node
        if feature_idx is None:
            counter = Counter(y)
            value = counter.most_common(1)[0][0]
            return Node(value=value)
            
        # Update feature importance
        self.feature_importance[feature_idx] += 1
            
        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Create and return decision node
        return Node(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        # Reset feature importance
        self.feature_importance = None
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Normalize feature importance
        if self.feature_importance is not None and np.sum(self.feature_importance) > 0:
            self.feature_importance = self.feature_importance / np.sum(self.feature_importance)
        
    def update(self, x: np.ndarray, y: int):
        """
        Update the model with a new sample.
        
        Note: Decision trees don't support true incremental updates,
        so this is a placeholder. For real updates, collect samples
        and call fit() again.
        
        Args:
            x: Feature vector
            y: Label
        """
        # Decision trees don't support true incremental updates
        # This is a placeholder
        pass
            
    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """
        Predict the class for a single sample.
        
        Args:
            x: Feature vector
            node: Current node
            
        Returns:
            int: Predicted class
        """
        # If leaf node, return value
        if node.value is not None:
            return node.value
            
        # Otherwise, navigate the tree
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
            
    def predict(self, x: np.ndarray) -> int:
        """
        Predict the class for a sample.
        
        Args:
            x: Feature vector
            
        Returns:
            int: Predicted class
        """
        if self.root is None:
            return 0  # Default to Banker if not trained
            
        return self._predict_sample(x, self.root)
            
    def _predict_proba_sample(self, x: np.ndarray, node: Node) -> Tuple[int, float]:
        """
        Predict class probabilities for a single sample.
        
        Args:
            x: Feature vector
            node: Current node
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        # If leaf node, return value with confidence 1.0
        if node.value is not None:
            return node.value, 1.0
            
        # Otherwise, navigate the tree
        if x[node.feature_idx] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        else:
            return self._predict_proba_sample(x, node.right)
            
    def predict_proba(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Predict class probabilities.
        
        Args:
            x: Feature vector
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if self.root is None:
            return 0, 0.5  # Default to Banker with neutral confidence
            
        # Decision trees don't naturally provide probabilities
        # We'll use a simple heuristic: confidence is 1.0 at leaf nodes
        return self._predict_proba_sample(x, self.root)
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            "algorithm": "Decision Tree",
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "n_features": self.n_features
        }
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on information gain.
        
        Returns:
            np.ndarray: Feature importance
        """
        if self.feature_importance is None:
            return np.array([])
            
        return self.feature_importance
