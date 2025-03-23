"""
Trend Confirmation strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TrendConfirmationStrategy(BaseStrategy):
    """
    Trend Confirmation strategy - looks for consensus across multiple timeframes
    before placing a bet, requiring strong trend confirmation.
    
    This strategy uses multiple sliding windows to analyze trends at different 
    timeframes, and only bets when there is agreement between them.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.window_sizes = self.params.get('window_sizes', [5, 10, 20])
        self.min_threshold = self.params.get('min_threshold', 0.6)
        self.confirmation_threshold = self.params.get('confirmation_threshold', 2)
        self.skip_enabled = self.params.get('skip_enabled', True)
        
        # Initialize state
        self._state = {
            'last_prediction': None,
            'win_streak': 0,
            'loss_streak': 0,
            'performance_history': []
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using trend confirmation across multiple timeframes
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < max(self.window_sizes):
            return 'B'  # Default when not enough history
        
        state = self._state
        
        # Update performance tracking
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            state['performance_history'].append(1 if is_win else 0)
            if len(state['performance_history']) > 20:  # Keep last 20 results
                state['performance_history'].pop(0)
                
            if is_win:
                state['win_streak'] += 1
                state['loss_streak'] = 0
            else:
                state['loss_streak'] += 1
                state['win_streak'] = 0
        
        # Analyze trends in multiple timeframes
        timeframe_votes = []
        timeframe_confidence = []
        
        for window_size in self.window_sizes:
            if len(historical_outcomes) >= window_size:
                recent = historical_outcomes[-window_size:]
                p_count = recent.count('P')
                b_count = recent.count('B')
                total = p_count + b_count
                
                if total > 0:
                    p_ratio = p_count / total
                    
                    # Calculate trend strength and direction
                    if p_ratio > 0.5:
                        # P trend
                        strength = p_ratio - 0.5  # 0 to 0.5
                        confidence = 0.5 + strength  # 0.5 to 1.0
                        if confidence >= self.min_threshold:
                            timeframe_votes.append('P')
                            timeframe_confidence.append(confidence)
                    else:
                        # B trend
                        strength = 0.5 - p_ratio  # 0 to 0.5
                        confidence = 0.5 + strength  # 0.5 to 1.0
                        if confidence >= self.min_threshold:
                            timeframe_votes.append('B')
                            timeframe_confidence.append(confidence)
        
        # Check for trend agreement
        p_votes = timeframe_votes.count('P')
        b_votes = timeframe_votes.count('B')
        
        # Calculate average confidence for each side
        p_confidence = sum([timeframe_confidence[i] for i, v in enumerate(timeframe_votes) if v == 'P'])
        p_confidence = p_confidence / p_votes if p_votes > 0 else 0
        
        b_confidence = sum([timeframe_confidence[i] for i, v in enumerate(timeframe_votes) if v == 'B'])
        b_confidence = b_confidence / b_votes if b_votes > 0 else 0
        
        # Apply banker bias (due to lower commission)
        b_confidence *= 1.02
        
        # Decision logic
        if p_votes >= self.confirmation_threshold and p_confidence > b_confidence:
            next_bet = 'P'
        elif b_votes >= self.confirmation_threshold and b_confidence > p_confidence:
            next_bet = 'B'
        else:
            # No clear trend confirmation - skip or use default
            if self.skip_enabled:
                next_bet = 'SKIP'
            else:
                next_bet = 'B'  # Default to banker due to lower commission
        
        # Special case - if we've lost several times in a row, be more cautious
        if state['loss_streak'] >= 3 and next_bet != 'SKIP' and self.skip_enabled:
            # Calculate overall win rate
            win_rate = sum(state['performance_history']) / len(state['performance_history']) if state['performance_history'] else 0.5
            
            # If we're doing poorly, skip more often
            if win_rate < 0.4:
                next_bet = 'SKIP'
        
        # Store prediction
        state['last_prediction'] = next_bet
        return next_bet