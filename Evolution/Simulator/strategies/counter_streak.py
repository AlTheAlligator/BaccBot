"""
Counter Streak strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class CounterStreakStrategy(BaseStrategy):
    """
    Bet against the side that's on a winning streak.
    This strategy looks for streaks of the same outcome and bets against continuation.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.streak_length = self.params.get('streak_length', 3)
    
    def get_bet(self, outcomes):
        """
        Determine next bet by countering streak patterns
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.streak_length:
            return 'P'  # Default when not enough history
        
        # Look at the last streak_length outcomes
        last_n = historical_outcomes[-self.streak_length:]
        
        # Verify if we have a streak (all same outcomes)
        first = last_n[0]
        is_streak = all(outcome == first for outcome in last_n)
        
        if is_streak:
            return 'P' if first == 'B' else 'B'  # Bet against the streak
        
        # No streak, bet opposite of last known outcome
        return 'P' if historical_outcomes[-1] == 'B' else 'B'