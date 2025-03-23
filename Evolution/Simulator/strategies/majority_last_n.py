"""
Majority Last N strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MajorityLastNStrategy(BaseStrategy):
    """
    Bet based on the majority in the last N outcomes.
    This strategy analyzes frequency of outcomes and bets against the majority.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.n = self.params.get('n', 6)
    
    def get_bet(self, outcomes):
        """
        Determine next bet based on majority in recent outcomes
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.n:
            return 'B'  # Default when not enough history
        
        # Get the last n outcomes
        last_n = historical_outcomes[-self.n:]
        p_count = last_n.count('P')
        b_count = last_n.count('B')
        
        # If there's a clear majority (more than 60%), bet against it
        total = p_count + b_count
        if total > 0:
            if p_count / total > 0.6:
                return 'B'
            elif b_count / total > 0.6:
                return 'P'
        
        # If no clear majority, bet on the less frequent outcome
        return 'P' if p_count <= b_count else 'B'