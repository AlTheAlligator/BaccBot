"""
Hybrid Ensemble strategy implementation.
"""

import logging
from collections import Counter, defaultdict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HybridEnsembleStrategy(BaseStrategy):
    """
    Advanced hybrid strategy that combines multiple statistical methods to create
    an ensemble approach for betting decisions.
    
    This strategy uses a weighted combination of:
    - Pattern detection
    - Multi-timeframe trend analysis 
    - Streak analysis
    
    Each component contributes to the final decision with configurable weights.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize ensemble strategy with parameter validation.
        
        Args:
            simulator: Reference to the GameSimulator instance
            params: Dictionary of strategy-specific parameters
        """
        super().__init__(simulator, params)
        
        # Validate and set window sizes
        self.window_sizes = self.params.get('window_sizes', [2, 1, 1])
        if not isinstance(self.window_sizes, list) or len(self.window_sizes) != 3:
            raise ValueError("window_sizes must be a list of 3 integers")
        if not all(isinstance(w, int) and w > 0 for w in self.window_sizes):
            raise ValueError("All window sizes must be positive integers")
            
        # Validate and set other parameters
        self.pattern_length = self.params.get('pattern_length', 3)
        if not isinstance(self.pattern_length, int) or self.pattern_length < 2:
            raise ValueError("pattern_length must be an integer >= 2")
            
        self.streak_threshold = self.params.get('streak_threshold', 4)
        if not isinstance(self.streak_threshold, int) or self.streak_threshold < 2:
            raise ValueError("streak_threshold must be an integer >= 2")
            
        self.pattern_weight = self.params.get('pattern_weight', 0.5)
        if not isinstance(self.pattern_weight, (int, float)) or not 0 <= self.pattern_weight <= 1:
            raise ValueError("pattern_weight must be a float between 0 and 1")
            
        self.confidence_threshold = self.params.get('confidence_threshold', 0.65)
        if not isinstance(self.confidence_threshold, (int, float)) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be a float between 0 and 1")
            
        # Initialize pattern tracking
        self._pattern_stats = defaultdict(lambda: {'P': 0, 'B': 0})
        
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tWindow sizes: {self.window_sizes}"
                   f"\n\tPattern length: {self.pattern_length}"
                   f"\n\tStreak threshold: {self.streak_threshold}"
                   f"\n\tPattern weight: {self.pattern_weight}"
                   f"\n\tConfidence threshold: {self.confidence_threshold}")

    def get_bet(self, outcomes):
        """
        Determine next bet using ensemble of strategies.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' for Player, 'B' for Banker, or 'SKIP' if confidence is too low
        """
        if not outcomes:
            return 'B'  # Default when no history
            
        historical = self._get_historical_outcomes(outcomes)
        if len(historical) < max(self.window_sizes):
            return 'B'  # Not enough history
            
        # Update pattern statistics
        if len(historical) >= self.pattern_length + 1:
            pattern = ''.join(historical[-(self.pattern_length+1):-1])
            outcome = historical[-1]
            self._pattern_stats[pattern][outcome] += 1
            
        # Calculate component votes
        pattern_vote = self._analyze_patterns(historical)
        trend_vote = self._analyze_trends(historical)
        streak_vote = self._analyze_streaks(historical)
        
        # Combine votes with weights
        pattern_weight = self.pattern_weight
        trend_weight = (1 - pattern_weight) * 0.6
        streak_weight = (1 - pattern_weight) * 0.4
        
        total_p = (pattern_vote[0] * pattern_weight + 
                  trend_vote[0] * trend_weight +
                  streak_vote[0] * streak_weight)
                  
        total_b = (pattern_vote[1] * pattern_weight + 
                  trend_vote[1] * trend_weight +
                  streak_vote[1] * streak_weight)
                  
        # Calculate confidence
        confidence = max(total_p, total_b) / (total_p + total_b) if (total_p + total_b) > 0 else 0.5
        
        if confidence < self.confidence_threshold:
            return 'SKIP'
            
        return 'P' if total_p > total_b else 'B'
        
    def _analyze_patterns(self, outcomes):
        """
        Analyze recent patterns more efficiently using stored pattern statistics.
        
        Returns:
            tuple: (p_probability, b_probability)
        """
        if len(outcomes) < self.pattern_length:
            return (0.5, 0.5)
            
        current_pattern = ''.join(outcomes[-self.pattern_length:])
        
        # Use stored pattern statistics
        stats = self._pattern_stats[current_pattern]
        total = stats['P'] + stats['B']
        
        if total >= 3:  # Require minimum observations
            p_prob = stats['P'] / total
            return (p_prob, 1 - p_prob)
            
        # Fallback to basic pattern matching for new patterns
        matches = []
        for i in range(len(outcomes) - self.pattern_length - 1):
            if ''.join(outcomes[i:i+self.pattern_length]) == current_pattern:
                matches.append(outcomes[i + self.pattern_length])
                
        if matches:
            p_prob = matches.count('P') / len(matches)
            return (p_prob, 1 - p_prob)
            
        return (0.5, 0.5)
        
    def _analyze_trends(self, outcomes):
        """
        Analyze trends across multiple timeframes more efficiently
        
        Returns:
            tuple: (p_probability, b_probability)
        """
        if len(outcomes) < max(self.window_sizes):
            return (0.5, 0.5)
            
        # Calculate all counts in one pass
        counts = Counter(outcomes[-max(self.window_sizes):])
        
        p_weights = []
        b_weights = []
        
        for window in sorted(self.window_sizes):
            window_outcomes = outcomes[-window:]
            p_count = window_outcomes.count('P')
            b_count = window_outcomes.count('B')
            total = p_count + b_count
            
            if total > 0:
                p_weights.append(p_count / total)
                b_weights.append(b_count / total)
                
        if not p_weights:
            return (0.5, 0.5)
            
        # Weight recent windows more heavily
        total_weight = sum(1.2 ** i for i in range(len(p_weights)))
        p_avg = sum((1.2 ** i) * w for i, w in enumerate(p_weights)) / total_weight
        
        return (p_avg, 1 - p_avg)
        
    def _analyze_streaks(self, outcomes):
        """
        Analyze current streaks and their significance
        
        Returns:
            tuple: (p_probability, b_probability)
        """
        if len(outcomes) < 2:
            return (0.5, 0.5)
            
        # Find current streak
        current_streak = 1
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == outcomes[-1]:
                current_streak += 1
            else:
                break
                
        if current_streak >= self.streak_threshold:
            # Long streaks tend to break
            return (0.7, 0.3) if outcomes[-1] == 'B' else (0.3, 0.7)
        else:
            # Short streaks tend to continue
            return (0.3, 0.7) if outcomes[-1] == 'B' else (0.7, 0.3)