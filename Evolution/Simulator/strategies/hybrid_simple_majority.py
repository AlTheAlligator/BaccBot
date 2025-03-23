"""
Hybrid Simple Majority strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class HybridSimpleMajorityStrategy(BaseStrategy):
    """
    Hybrid Simple Majority strategy - uses simple voting with equal weights.
    
    This is a simplified version of the HybridMajorityStrategy that uses
    equal weighting for all signals rather than configurable weights.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.window_size = self.params.get('window_size', 15)
        self.confidence_threshold = self.params.get('confidence_threshold', 0.6)
        
    def get_bet(self, outcomes):
        """
        This strategy is a hybrid approach that combines multiple signals
        with equal weights. The integration is handled in the simulator.
        
        Returns None to indicate this should be handled by the simulator.
        """
        return None