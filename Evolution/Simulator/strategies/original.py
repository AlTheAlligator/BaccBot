"""
Original strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class OriginalStrategy(BaseStrategy):
    """
    Original BBB/PPP strategy - bets on the opposite of the current mode.
    
    This strategy implements the classic BBB/PPP betting pattern where we bet against
    the current mode and switch modes when encountering 3 consecutive losses.
    
    The strategy handles ties by treating them as non-events, preserving the current
    betting pattern and mode.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        # For the original strategy, we rely on the simulator's mode switching logic
        self._validate_mode(simulator.current_mode)
        self._validate_mode(simulator.initial_mode)
        if simulator.initial_mode not in ["PPP", "BBB"]:
            raise ValueError("initial_mode must be 'PPP' or 'BBB'")
            
        # Disable outcome validation for performance since this strategy is called frequently
        self.validate_outcomes = False
        
    def _validate_mode(self, mode):
        """Validate that the mode is supported by this strategy"""
        valid_modes = {"BBB", "PPP", "Switch"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {valid_modes}")
    
    def get_bet(self, outcomes):
        """
        Return bet based on the original BBB/PPP strategy.
        
        Args:
            outcomes: List of historical outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' for Player, 'B' for Banker
            
        Note:
            This strategy always returns a bet (never SKIP) as it's designed 
            to bet on every hand. Ties are handled gracefully by preserving 
            the current mode and betting pattern.
        """
        if not isinstance(outcomes, list):
            raise TypeError("outcomes must be a list")
            
        # Validate current mode
        self._validate_mode(self.simulator.current_mode)
        
        # Get the actual bet based on current mode
        if self.simulator.current_mode == "BBB":
            return "P"  # If we're in BBB mode, we bet on Player
        elif self.simulator.current_mode == "PPP":
            return "B"  # If we're in PPP mode, we bet on Banker
        else:  # Switch mode - bet opposite of last bet
            if not self.simulator.last_bet:
                # If no last bet (first bet in switch mode), use opposite of initial mode
                return "B" if self.simulator.initial_mode == "PPP" else "P"
            return "B" if self.simulator.last_bet.side == "P" else "P"