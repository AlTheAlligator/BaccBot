"""
Ensemble Algorithm for Baccarat Prediction

This module implements an ensemble of multiple ML algorithms for predicting baccarat outcomes.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class EnsemblePredictor:
    """
    Ensemble predictor for baccarat outcomes.
    
    Features:
    - Combines multiple ML algorithms
    - Weighted voting
    - Confidence-based decision making
    - Feature importance aggregation
    """
    
    def __init__(self, predictors=None, weights=None):
        """
        Initialize the ensemble predictor.
        
        Args:
            predictors: List of predictor objects
            weights: List of weights for each predictor
        """
        self.predictors = predictors or []
        
        # Initialize equal weights if not provided
        if weights is None:
            self.weights = np.ones(len(self.predictors)) / len(self.predictors) if self.predictors else np.array([])
        else:
            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights) if np.sum(weights) > 0 else np.array(weights)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all predictors in the ensemble.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        for predictor in self.predictors:
            predictor.fit(X, y)
        
    def update(self, x: np.ndarray, y: int):
        """
        Update all predictors with a new sample.
        
        Args:
            x: Feature vector
            y: Label
        """
        for predictor in self.predictors:
            predictor.update(x, y)
            
    def predict(self, x: np.ndarray) -> int:
        """
        Predict the class for a sample using weighted voting.
        
        Args:
            x: Feature vector
            
        Returns:
            int: Predicted class
        """
        if not self.predictors:
            return 0  # Default to Banker if no predictors
            
        # Get predictions from all predictors
        predictions = []
        for predictor in self.predictors:
            predictions.append(predictor.predict(x))
            
        # Count weighted votes
        player_votes = 0
        banker_votes = 0
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # Player
                player_votes += self.weights[i]
            else:  # Banker
                banker_votes += self.weights[i]
                
        return 1 if player_votes > banker_votes else 0
            
    def predict_proba(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Predict class probabilities using weighted confidence.
        
        Args:
            x: Feature vector
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if not self.predictors:
            return 0, 0.5  # Default to Banker with neutral confidence
            
        # Get predictions and confidences from all predictors
        predictions = []
        confidences = []
        
        for predictor in self.predictors:
            pred, conf = predictor.predict_proba(x)
            predictions.append(pred)
            confidences.append(conf)
            
        # Calculate weighted confidence for each class
        player_confidence = 0
        banker_confidence = 0
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if pred == 1:  # Player
                player_confidence += self.weights[i] * conf
            else:  # Banker
                banker_confidence += self.weights[i] * conf
                
        # Normalize confidences
        total_confidence = player_confidence + banker_confidence
        
        if total_confidence > 0:
            player_confidence /= total_confidence
            banker_confidence /= total_confidence
            
        # Return class with highest confidence
        if player_confidence > banker_confidence:
            return 1, player_confidence
        else:
            return 0, banker_confidence
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        predictor_params = []
        for i, predictor in enumerate(self.predictors):
            params = predictor.get_params()
            params["weight"] = float(self.weights[i]) if i < len(self.weights) else 0
            predictor_params.append(params)
            
        return {
            "algorithm": "Ensemble",
            "n_predictors": len(self.predictors),
            "predictors": predictor_params
        }
        
    def get_feature_importance(self) -> np.ndarray:
        """
        Get aggregated feature importance from all predictors.
        
        Returns:
            np.ndarray: Feature importance
        """
        if not self.predictors:
            return np.array([])
            
        # Get feature importance from each predictor
        importances = []
        
        for i, predictor in enumerate(self.predictors):
            if hasattr(predictor, 'get_feature_importance'):
                importance = predictor.get_feature_importance()
                if len(importance) > 0:
                    # Weight by predictor weight
                    weight = self.weights[i] if i < len(self.weights) else 0
                    importances.append(importance * weight)
                    
        if not importances:
            return np.array([])
            
        # Ensure all importance arrays have the same length
        max_length = max(len(imp) for imp in importances)
        padded_importances = []
        
        for imp in importances:
            if len(imp) < max_length:
                # Pad with zeros
                padded = np.zeros(max_length)
                padded[:len(imp)] = imp
                padded_importances.append(padded)
            else:
                padded_importances.append(imp)
                
        # Aggregate importances
        aggregated = np.sum(padded_importances, axis=0)
        
        # Normalize
        if np.sum(aggregated) > 0:
            aggregated = aggregated / np.sum(aggregated)
            
        return aggregated
