"""
Feature Engineering for Machine Learning Strategies

This module provides feature extraction and engineering for ML-based baccarat strategies.
"""

import numpy as np
from collections import Counter, deque
from typing import List, Dict, Any, Tuple

class FeatureEngineer:
    """
    Feature engineering for baccarat ML strategies.
    
    Extracts various features from outcome history:
    - Pattern-based features (n-grams)
    - Frequency-based features
    - Streak-based features
    - Statistical features
    """
    
    def __init__(self, use_pattern_features=True, use_frequency_features=True,
                use_streak_features=True, use_statistical_features=True,
                pattern_length=3, window_size=10):
        """
        Initialize the feature engineer.
        
        Args:
            use_pattern_features: Whether to use pattern-based features
            use_frequency_features: Whether to use frequency-based features
            use_streak_features: Whether to use streak-based features
            use_statistical_features: Whether to use statistical features
            pattern_length: Length of patterns to extract
            window_size: Size of window for feature extraction
        """
        self.use_pattern_features = use_pattern_features
        self.use_frequency_features = use_frequency_features
        self.use_streak_features = use_streak_features
        self.use_statistical_features = use_statistical_features
        self.pattern_length = pattern_length
        self.window_size = window_size
        
        # Initialize feature names
        self._init_feature_names()
        
    def _init_feature_names(self):
        """Initialize feature names for reference."""
        self.feature_names = []
        
        # Pattern features
        if self.use_pattern_features:
            for i in range(self.pattern_length):
                self.feature_names.append(f"pattern_{i}")
                
        # Frequency features
        if self.use_frequency_features:
            self.feature_names.extend([
                "player_freq_short",
                "player_freq_medium",
                "player_freq_long",
                "banker_freq_short",
                "banker_freq_medium",
                "banker_freq_long"
            ])
            
        # Streak features
        if self.use_streak_features:
            self.feature_names.extend([
                "current_streak_type",
                "current_streak_length",
                "max_player_streak",
                "max_banker_streak"
            ])
            
        # Statistical features
        if self.use_statistical_features:
            self.feature_names.extend([
                "alternating_rate",
                "player_after_player",
                "banker_after_banker",
                "player_after_banker",
                "banker_after_player",
                "entropy"
            ])
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of extracted features.
        
        Returns:
            list: Feature names
        """
        return self.feature_names
    
    def _outcomes_to_numeric(self, outcomes: List[str]) -> List[int]:
        """
        Convert outcome strings to numeric values.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            list: Numeric outcomes (1 for P, 0 for B, -1 for T)
        """
        return [1 if o == 'P' else (0 if o == 'B' else -1) for o in outcomes]
    
    def _extract_pattern_features(self, numeric_outcomes: List[int]) -> List[float]:
        """
        Extract pattern-based features.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            list: Pattern features
        """
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        # Get the most recent pattern
        pattern = filtered[-self.pattern_length:] if len(filtered) >= self.pattern_length else []
        
        # Pad pattern if needed
        pattern = pattern + [0] * (self.pattern_length - len(pattern))
        
        return pattern
    
    def _extract_frequency_features(self, numeric_outcomes: List[int]) -> List[float]:
        """
        Extract frequency-based features.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            list: Frequency features
        """
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        # Define window sizes
        short_window = min(len(filtered), self.window_size // 2)
        medium_window = min(len(filtered), self.window_size)
        long_window = min(len(filtered), self.window_size * 2)
        
        # Calculate player frequencies
        player_freq_short = sum(filtered[-short_window:]) / short_window if short_window > 0 else 0.5
        player_freq_medium = sum(filtered[-medium_window:]) / medium_window if medium_window > 0 else 0.5
        player_freq_long = sum(filtered[-long_window:]) / long_window if long_window > 0 else 0.5
        
        # Calculate banker frequencies
        banker_freq_short = 1 - player_freq_short
        banker_freq_medium = 1 - player_freq_medium
        banker_freq_long = 1 - player_freq_long
        
        return [
            player_freq_short,
            player_freq_medium,
            player_freq_long,
            banker_freq_short,
            banker_freq_medium,
            banker_freq_long
        ]
    
    def _extract_streak_features(self, numeric_outcomes: List[int]) -> List[float]:
        """
        Extract streak-based features.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            list: Streak features
        """
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if not filtered:
            return [0, 0, 0, 0]  # Default values
            
        # Calculate current streak
        current_streak_type = filtered[-1]  # 1 for Player, 0 for Banker
        current_streak_length = 1
        
        for i in range(len(filtered) - 2, -1, -1):
            if filtered[i] == current_streak_type:
                current_streak_length += 1
            else:
                break
                
        # Calculate max streaks
        max_player_streak = 0
        max_banker_streak = 0
        current_streak = 1
        
        for i in range(1, len(filtered)):
            if filtered[i] == filtered[i-1]:
                current_streak += 1
            else:
                if filtered[i-1] == 1:  # Player
                    max_player_streak = max(max_player_streak, current_streak)
                else:  # Banker
                    max_banker_streak = max(max_banker_streak, current_streak)
                current_streak = 1
                
        # Check last streak
        if filtered[-1] == 1:  # Player
            max_player_streak = max(max_player_streak, current_streak)
        else:  # Banker
            max_banker_streak = max(max_banker_streak, current_streak)
            
        # Normalize streak lengths
        max_streak = max(1, max(max_player_streak, max_banker_streak))
        current_streak_length_norm = current_streak_length / max_streak
        max_player_streak_norm = max_player_streak / max_streak
        max_banker_streak_norm = max_banker_streak / max_streak
        
        return [
            current_streak_type,
            current_streak_length_norm,
            max_player_streak_norm,
            max_banker_streak_norm
        ]
    
    def _extract_statistical_features(self, numeric_outcomes: List[int]) -> List[float]:
        """
        Extract statistical features.
        
        Args:
            numeric_outcomes: List of numeric outcomes
            
        Returns:
            list: Statistical features
        """
        # Filter out ties
        filtered = [o for o in numeric_outcomes if o != -1]
        
        if len(filtered) < 2:
            return [0, 0, 0, 0, 0, 0]  # Default values
            
        # Calculate alternating rate
        alternating_count = 0
        for i in range(1, len(filtered)):
            if filtered[i] != filtered[i-1]:
                alternating_count += 1
                
        alternating_rate = alternating_count / (len(filtered) - 1)
        
        # Calculate transition probabilities
        transitions = {
            "PP": 0,  # Player after Player
            "BB": 0,  # Banker after Banker
            "PB": 0,  # Banker after Player
            "BP": 0   # Player after Banker
        }
        
        p_count = 0
        b_count = 0
        
        for i in range(1, len(filtered)):
            prev = filtered[i-1]
            curr = filtered[i]
            
            if prev == 1:  # Player
                p_count += 1
                if curr == 1:  # Player after Player
                    transitions["PP"] += 1
                else:  # Banker after Player
                    transitions["PB"] += 1
            else:  # Banker
                b_count += 1
                if curr == 1:  # Player after Banker
                    transitions["BP"] += 1
                else:  # Banker after Banker
                    transitions["BB"] += 1
                    
        # Calculate conditional probabilities
        player_after_player = transitions["PP"] / p_count if p_count > 0 else 0.5
        banker_after_banker = transitions["BB"] / b_count if b_count > 0 else 0.5
        player_after_banker = transitions["BP"] / b_count if b_count > 0 else 0.5
        banker_after_player = transitions["PB"] / p_count if p_count > 0 else 0.5
        
        # Calculate entropy
        p_freq = sum(filtered) / len(filtered)
        b_freq = 1 - p_freq
        
        entropy = 0
        if p_freq > 0:
            entropy -= p_freq * np.log2(p_freq)
        if b_freq > 0:
            entropy -= b_freq * np.log2(b_freq)
            
        # Normalize entropy (max entropy is 1 for binary outcomes)
        entropy = entropy / 1.0 if entropy > 0 else 0
        
        return [
            alternating_rate,
            player_after_player,
            banker_after_banker,
            player_after_banker,
            banker_after_player,
            entropy
        ]
    
    def extract_features(self, outcomes: List[str]) -> np.ndarray:
        """
        Extract all features from outcome history.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            np.ndarray: Feature vector
        """
        # Convert outcomes to numeric
        numeric = self._outcomes_to_numeric(outcomes)
        
        # Initialize feature list
        features = []
        
        # Extract pattern features
        if self.use_pattern_features:
            pattern_features = self._extract_pattern_features(numeric)
            features.extend(pattern_features)
            
        # Extract frequency features
        if self.use_frequency_features:
            freq_features = self._extract_frequency_features(numeric)
            features.extend(freq_features)
            
        # Extract streak features
        if self.use_streak_features:
            streak_features = self._extract_streak_features(numeric)
            features.extend(streak_features)
            
        # Extract statistical features
        if self.use_statistical_features:
            stat_features = self._extract_statistical_features(numeric)
            features.extend(stat_features)
            
        return np.array(features)
