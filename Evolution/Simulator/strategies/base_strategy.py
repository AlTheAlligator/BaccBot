"""
Base Strategy class definition.
"""

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all baccarat betting strategies.
    
    All specific strategy implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the base strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy (default: {})
        """
        if simulator is None:
            raise ValueError("simulator cannot be None")
            
        if params is not None and not isinstance(params, dict):
            raise ValueError("params must be a dictionary if provided")
            
        self.simulator = simulator
        self.params = params if params else {}
        self.validate_outcomes = True  # Can be disabled for performance in child classes
        logger.debug(f"Initialized {self.__class__.__name__} with params: {self.params}")
    
    def _validate_outcome_list(self, outcomes):
        """
        Validate that outcomes list contains only valid values.
        
        Args:
            outcomes: List of outcomes to validate
            
        Raises:
            ValueError: If outcomes contains invalid values
            TypeError: If outcomes is not a list
        """
        if not isinstance(outcomes, list):
            raise TypeError("outcomes must be a list")
            
        if self.validate_outcomes:
            if any(o not in ['P', 'B', 'T'] for o in outcomes):
                raise ValueError("outcomes must only contain 'P', 'B', or 'T'")

    @abstractmethod
    def get_bet(self, outcomes):
        """
        Get the next bet based on current outcomes.
        
        Args:
            outcomes: List of current outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' for Player, 'B' for Banker, or 'SKIP' to not place a bet
        """
        self._validate_outcome_list(outcomes)
        pass
        
    def _get_historical_outcomes(self, outcomes):
        """
        Get list of historical outcomes excluding ties.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            list: Historical outcomes without ties
        """
        return [o for o in outcomes if o in ('P', 'B')]
        
    def _analyze_window(self, outcomes, window_size):
        """
        Analyze outcomes in a specific window.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            window_size: Size of window to analyze
            
        Returns:
            tuple: (p_ratio, b_ratio, total_count)
        """
        if len(outcomes) < window_size:
            return 0.5, 0.5, 0
            
        window = outcomes[-window_size:]
        counts = Counter(window)
        total = counts['P'] + counts['B']
        
        if total == 0:
            return 0.5, 0.5, 0
            
        p_ratio = counts['P'] / total
        return p_ratio, 1 - p_ratio, total
        
    def _get_current_streak(self, outcomes):
        """
        Get the current streak length and value.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            tuple: (streak_value, streak_length)
        """
        if not outcomes:
            return None, 0
            
        historical = self._get_historical_outcomes(outcomes)
        if not historical:
            return None, 0
            
        streak_value = historical[-1]
        streak_length = 1
        
        for i in range(len(historical)-2, -1, -1):
            if historical[i] == streak_value:
                streak_length += 1
            else:
                break
                
        return streak_value, streak_length
        
    def _analyze_pattern(self, outcomes, pattern_length, min_matches=3):
        """
        Analyze pattern occurrence and following outcomes.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            pattern_length: Length of pattern to look for
            min_matches: Minimum number of pattern matches required
            
        Returns:
            tuple: (pattern_found, p_probability, total_matches)
        """
        if len(outcomes) < pattern_length * 2:
            return False, 0.5, 0
            
        historical = self._get_historical_outcomes(outcomes)
        if len(historical) < pattern_length * 2:
            return False, 0.5, 0
            
        current_pattern = historical[-pattern_length:]
        matches = []
        
        for i in range(len(historical) - pattern_length * 2):
            test_pattern = historical[i:i+pattern_length]
            if test_pattern == current_pattern and i + pattern_length < len(historical):
                matches.append(historical[i + pattern_length])
                
        if len(matches) < min_matches:
            return False, 0.5, len(matches)
            
        p_probability = matches.count('P') / len(matches)
        return True, p_probability, len(matches)
        
    def _is_alternating(self, outcomes, length=4):
        """
        Check if recent outcomes are alternating.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            length: Number of outcomes to check for alternating pattern
            
        Returns:
            tuple: (is_alternating, last_value)
        """
        historical = self._get_historical_outcomes(outcomes)
        if len(historical) < length:
            return False, None
            
        recent = historical[-length:]
        alternating = True
        
        for i in range(1, len(recent)):
            if recent[i] == recent[i-1]:
                alternating = False
                break
                
        return alternating, recent[-1] if alternating else None