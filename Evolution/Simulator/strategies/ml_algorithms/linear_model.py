"""
Linear Model Algorithm for Baccarat Prediction

This module implements linear models (logistic regression, linear regression)
for predicting baccarat outcomes.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class LinearModelPredictor:
    """
    Linear model predictor for baccarat outcomes.
    
    Features:
    - Multiple model types (logistic regression, linear regression)
    - L2 regularization
    - Incremental updates with stochastic gradient descent
    - Feature importance estimation
    """
    
    def __init__(self, model_type='logistic', learning_rate=0.01, regularization=0.01, max_iter=1000):
        """
        Initialize the linear model predictor.
        
        Args:
            model_type: Type of model ('logistic' or 'linear')
            learning_rate: Learning rate for gradient descent
            regularization: L2 regularization strength
            max_iter: Maximum number of iterations for batch training
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0.0
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        Args:
            z: Input values
            
        Returns:
            np.ndarray: Sigmoid output
        """
        # Clip to avoid overflow
        z = np.clip(z, -100, 100)
        return 1.0 / (1.0 + np.exp(-z))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear model using gradient descent.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        n_samples, n_features = X.shape
        
        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            
        # Gradient descent
        for _ in range(self.max_iter):
            # Forward pass
            if self.model_type == 'logistic':
                y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            else:  # linear
                y_pred = np.dot(X, self.weights) + self.bias
                
            # Compute gradients
            if self.model_type == 'logistic':
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + (self.regularization * self.weights)
                db = (1/n_samples) * np.sum(y_pred - y)
            else:  # linear
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + (self.regularization * self.weights)
                db = (1/n_samples) * np.sum(y_pred - y)
                
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def update(self, x: np.ndarray, y: int):
        """
        Update the model with a new sample using stochastic gradient descent.
        
        Args:
            x: Feature vector
            y: Label
        """
        if self.weights is None:
            self.weights = np.zeros(x.shape[0])
            self.bias = 0.0
            
        # Forward pass
        if self.model_type == 'logistic':
            y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias)
        else:  # linear
            y_pred = np.dot(x, self.weights) + self.bias
            
        # Compute gradients
        dw = (y_pred - y) * x + (self.regularization * self.weights)
        db = y_pred - y
            
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
            
    def predict(self, x: np.ndarray) -> int:
        """
        Predict the class for a sample.
        
        Args:
            x: Feature vector
            
        Returns:
            int: Predicted class
        """
        if self.weights is None:
            return 0  # Default to Banker if not trained
            
        # Forward pass
        if self.model_type == 'logistic':
            y_pred = self._sigmoid(np.dot(x, self.weights) + self.bias)
        else:  # linear
            y_pred = np.dot(x, self.weights) + self.bias
            
        # Classification threshold
        return 1 if y_pred >= 0.5 else 0
            
    def predict_proba(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Predict class probabilities.
        
        Args:
            x: Feature vector
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if self.weights is None:
            return 0, 0.5  # Default to Banker with neutral confidence
            
        # Forward pass
        if self.model_type == 'logistic':
            prob = self._sigmoid(np.dot(x, self.weights) + self.bias)
        else:  # linear
            # Clip linear output to [0, 1] range
            prob = np.clip(np.dot(x, self.weights) + self.bias, 0, 1)
            
        # Determine class and confidence
        if prob >= 0.5:
            return 1, prob
        else:
            return 0, 1 - prob
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            "algorithm": f"Linear ({self.model_type})",
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "n_features": len(self.weights) if self.weights is not None else 0
        }
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on coefficient magnitudes.
        
        Returns:
            np.ndarray: Feature importance
        """
        if self.weights is None:
            return np.array([])
            
        # Use absolute values of coefficients as importance
        importance = np.abs(self.weights)
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
            
        return importance
