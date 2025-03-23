"""
Streak Reversal Safe Exit strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class StreakReversalSafeExitStrategy(BaseStrategy):
    """
    Streak Reversal with Safe Exit strategy - bets on every outcome but uses specific
    rules to minimize consecutive losses and safely exit losing positions.
    
    This strategy always places a bet but adapts to patterns to minimize risks.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.streak_threshold = self.params.get('streak_threshold', 2)
        self.exit_loss_threshold = self.params.get('exit_loss_threshold', 2)
        self.recovery_period = self.params.get('recovery_period', 3)
        self.banker_bias = self.params.get('banker_bias', 0.51) # Default banker preference
        self.window_size = self.params.get('window_size', 15)
        
        # Initialize state
        self._state = {
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'current_side': 'B',  # Start with banker due to lower house edge
            'in_recovery_mode': False,
            'recovery_count': 0,
            'last_switch_reason': None,
            'switch_reason_performance': {},
            'pattern_stats': {},
            'last_prediction': None
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using streak reversal with safe exit
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < 3:
            return 'B'  # Default when not enough history
        
        state = self._state
        
        # Update state based on last prediction result
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            if is_win:
                state['consecutive_wins'] += 1
                state['consecutive_losses'] = 0
                
                # Track performance of switch reason
                if state['last_switch_reason']:
                    reason = state['last_switch_reason']
                    if reason not in state['switch_reason_performance']:
                        state['switch_reason_performance'][reason] = {'wins': 0, 'losses': 0}
                    state['switch_reason_performance'][reason]['wins'] += 1
            else:
                state['consecutive_losses'] += 1
                state['consecutive_wins'] = 0
                
                # Track performance of switch reason
                if state['last_switch_reason']:
                    reason = state['last_switch_reason']
                    if reason not in state['switch_reason_performance']:
                        state['switch_reason_performance'][reason] = {'wins': 0, 'losses': 0}
                    state['switch_reason_performance'][reason]['losses'] += 1
        
        # Update recovery mode status
        if state['consecutive_losses'] >= self.exit_loss_threshold:
            state['in_recovery_mode'] = True
            state['recovery_count'] = self.recovery_period
        elif state['in_recovery_mode']:
            state['recovery_count'] -= 1
            if state['recovery_count'] <= 0:
                state['in_recovery_mode'] = False
        
        # Generate pattern signature for tracking
        pattern_signature = None
        if len(historical_outcomes) >= 4:
            pattern_signature = ''.join(historical_outcomes[-4:])
            
            # Update pattern stats if we have previous result
            if state['last_prediction'] and len(outcomes) > 1:
                actual = outcomes[-2]
                
                # Only track pattern stats for Player and Banker outcomes (skip ties)
                if actual in ['P', 'B']:  
                    if pattern_signature not in state['pattern_stats']:
                        state['pattern_stats'][pattern_signature] = {'P': 0, 'B': 0}
                    
                    state['pattern_stats'][pattern_signature][actual] += 1
        
        # Decision Logic
        next_bet = None
        switch_reason = None
        
        # 1. Check for streaks in recent outcomes
        if len(historical_outcomes) >= self.streak_threshold:
            last_value = historical_outcomes[-1]
            streak_length = 1
            
            for i in range(len(historical_outcomes)-2, -1, -1):
                if historical_outcomes[i] == last_value:
                    streak_length += 1
                else:
                    break
            
            if streak_length >= self.streak_threshold:
                # Reverse bet against streak
                next_bet = 'P' if last_value == 'B' else 'B'
                switch_reason = f"streak_{last_value}_{streak_length}"
        
        # 2. Check for alternating pattern
        if next_bet is None and len(historical_outcomes) >= 3:
            if (historical_outcomes[-1] != historical_outcomes[-2] and 
                historical_outcomes[-2] != historical_outcomes[-3]):
                # Continue alternating pattern
                next_bet = 'P' if historical_outcomes[-1] == 'B' else 'B'
                switch_reason = "alternating_pattern"
        
        # 3. Check for pattern statistics if we have enough history
        if next_bet is None and pattern_signature is not None and pattern_signature in state['pattern_stats']:
            stats = state['pattern_stats'][pattern_signature]
            total = stats['P'] + stats['B']
            
            if total >= 3:  # Minimum observations
                p_ratio = stats['P'] / total if total > 0 else 0.5
                
                if p_ratio >= 0.7:
                    next_bet = 'P'
                    switch_reason = f"pattern_{pattern_signature}_P"
                elif p_ratio <= 0.3:
                    next_bet = 'B'
                    switch_reason = f"pattern_{pattern_signature}_B"
        
        # 4. Use switch reason performance statistics
        if next_bet is None and state['switch_reason_performance']:
            # Find the best performing switch reason
            best_reason = None
            best_win_ratio = 0
            
            for reason, stats in state['switch_reason_performance'].items():
                total = stats['wins'] + stats['losses']
                if total >= 3:  # Minimum observations
                    win_ratio = stats['wins'] / total if total > 0 else 0
                    if win_ratio > best_win_ratio:
                        best_win_ratio = win_ratio
                        best_reason = reason
            
            if best_reason and best_win_ratio >= 0.6:
                if 'streak_P' in best_reason:
                    next_bet = 'B'  # Counter Player streak
                elif 'streak_B' in best_reason:
                    next_bet = 'P'  # Counter Banker streak
                elif best_reason == 'alternating_pattern':
                    next_bet = 'P' if historical_outcomes[-1] == 'B' else 'B'
                
                switch_reason = f"best_reason_{best_reason}"
        
        # 5. For recovery mode - use safer strategies
        if state['in_recovery_mode']:
            # If in recovery mode, bias toward banker (lower house edge)
            if next_bet is None or state['consecutive_losses'] >= 3:
                next_bet = 'B'
                switch_reason = "recovery_mode_banker_bias"
            
            # If we've lost multiple times betting the same side, switch
            if state['consecutive_losses'] >= 3 and state['last_prediction'] == next_bet:
                next_bet = 'P' if next_bet == 'B' else 'B'
                switch_reason = "forced_side_change"
        
        # 6. Default decision if nothing else triggered
        if next_bet is None:
            # Slight banker bias due to lower commission
            if historical_outcomes[-1] == 'P':
                next_bet = 'B'
                switch_reason = "default_after_player"
            elif historical_outcomes[-1] == 'B':
                # Random selection with banker bias
                next_bet = 'B' if self.banker_bias >= 0.5 else 'P'
                switch_reason = "default_after_banker"
            else:
                next_bet = 'B'  # Default to banker on tie
                switch_reason = "default_after_tie"
        
        # Store the reason for switching for performance tracking
        state['last_switch_reason'] = switch_reason
        state['last_prediction'] = next_bet
        
        # Always return a legitimate bet (never SKIP)
        return next_bet