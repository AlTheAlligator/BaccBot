"""
Follow Streak strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class FollowStreakStrategy(BaseStrategy):
    """
    Follow Streak strategy - bets on the same side that's on a winning streak.
    
    This strategy looks for streaks of a certain length and then bets on 
    the same outcome, expecting the streak to continue.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.streak_length = self.params.get('streak_length', 3)
    
    def get_bet(self, outcomes):
        """
        Determine next bet by following streaks.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' for Player, 'B' for Banker
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        streak_length = self.params.get('streak_length', 3)
        
        if len(historical_outcomes) < streak_length:
            return 'B'  # Default when not enough history
        
        # Look for a streak of the specified length
        streak_value = historical_outcomes[-1]
        current_streak = 1
        
        for i in range(len(historical_outcomes)-2, -1, -1):
            if historical_outcomes[i] == streak_value:
                current_streak += 1
            else:
                break
                
            if current_streak >= streak_length:
                # Found a streak of the required length, follow it
                logger.debug(f"FOLLOW_STREAK: Found streak of {streak_value} with length {current_streak}")
                return streak_value
        
        # No streak found, default to banker (lower house edge)
        return 'B'