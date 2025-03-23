"""
Conservative Pattern strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ConservativePatternStrategy(BaseStrategy):
    """
    Conservative Pattern strategy - combines pattern matching with risk management.
    
    This strategy places bets only when pattern recognition achieves high confidence
    levels, and focuses on capital preservation during losing streaks.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.min_confidence = self.params.get('min_confidence', 0.7)
        self.pattern_length = self.params.get('pattern_length', 4)
        self.recovery_threshold = self.params.get('recovery_threshold', 2)
        self.skip_enabled = self.params.get('skip_enabled', True)
        
        # Initialize state
        self._state = {
            'last_prediction': None,
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'pattern_stats': {},
            'performance_window': []
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using conservative pattern approach
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.pattern_length * 2:
            return 'B'  # Default when not enough history
        
        state = self._state
        
        # Update performance tracking
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            state['performance_window'].append(1 if is_win else 0)
            if len(state['performance_window']) > 20:  # Keep last 20 results
                state['performance_window'].pop(0)
            
            if is_win:
                state['consecutive_wins'] += 1
                state['consecutive_losses'] = 0
            else:
                state['consecutive_losses'] += 1
                state['consecutive_wins'] = 0
            
            # Update pattern stats
            pattern = ''.join(historical_outcomes[-(self.pattern_length+1):-1])
            if pattern not in state['pattern_stats']:
                state['pattern_stats'][pattern] = {'P': 0, 'B': 0}
            state['pattern_stats'][pattern][actual] += 1
        
        # Calculate current pattern
        current_pattern = ''.join(historical_outcomes[-self.pattern_length:])
        
        # Pattern matching logic
        confidence = {'P': 0.5, 'B': 0.5}  # Starting confidence values
        
        # 1. Check if we have stats for this exact pattern
        if current_pattern in state['pattern_stats']:
            stats = state['pattern_stats'][current_pattern]
            total = stats['P'] + stats['B']
            
            if total >= 3:  # Need minimum samples to be confident
                p_ratio = stats['P'] / total
                
                pattern_boost = min(0.2, 0.05 * total)  # More samples, more confidence
                
                if p_ratio > 0.67:
                    confidence['P'] += pattern_boost
                elif p_ratio < 0.33:
                    confidence['B'] += pattern_boost
        
        # 2. Find similar patterns in history
        matches = []
        for i in range(len(historical_outcomes) - self.pattern_length * 2):
            past = historical_outcomes[i:i+self.pattern_length]
            if past == current_pattern and i + self.pattern_length < len(historical_outcomes):
                matches.append(historical_outcomes[i + self.pattern_length])
        
        if len(matches) >= 3:
            p_count = matches.count('P')
            p_ratio = p_count / len(matches)
            
            # Confidence boost based on number of matches
            match_boost = min(0.25, 0.05 * len(matches))
            
            if p_ratio > 0.67:
                confidence['P'] += match_boost
            elif p_ratio < 0.33:
                confidence['B'] += match_boost
        
        # 3. Additional confidence from streak analysis
        last = historical_outcomes[-1]
        streak = 1
        for i in range(len(historical_outcomes)-2, -1, -1):
            if historical_outcomes[i] == last:
                streak += 1
            else:
                break
                
        if streak >= 3:
            streak_boost = min(0.15, 0.03 * streak)
            # Bet against long streaks (mean reversion)
            if last == 'P':
                confidence['B'] += streak_boost
            else:
                confidence['P'] += streak_boost
        
        # 4. Alternating pattern detection
        if len(historical_outcomes) >= 4:
            alternating = True
            for i in range(1, min(4, len(historical_outcomes))):
                if historical_outcomes[-i] == historical_outcomes[-(i+1)]:
                    alternating = False
                    break
            
            if alternating:
                alt_boost = 0.15
                if historical_outcomes[-1] == 'P':
                    confidence['B'] += alt_boost
                else:
                    confidence['P'] += alt_boost
        
        # Apply banker edge due to commission
        confidence['B'] *= 1.02
        
        # Decision logic
        in_recovery_mode = state['consecutive_losses'] >= self.recovery_threshold
        current_min_confidence = self.min_confidence
        
        # In recovery mode, require higher confidence
        if in_recovery_mode:
            current_min_confidence += min(0.2, state['consecutive_losses'] * 0.03)
        
        # Check if win rate is poor
        if len(state['performance_window']) >= 10:
            win_rate = sum(state['performance_window']) / len(state['performance_window'])
            if win_rate < 0.4:
                current_min_confidence += 0.05  # Be more selective
        
        # Make decision based on confidence threshold
        p_confidence = confidence['P']
        b_confidence = confidence['B']
        
        if max(p_confidence, b_confidence) < current_min_confidence:
            if self.skip_enabled:
                next_bet = 'SKIP'
            else:
                next_bet = 'B'  # Default to banker
        elif p_confidence > b_confidence:
            next_bet = 'P'
        else:
            next_bet = 'B'
            
        # Store prediction
        state['last_prediction'] = next_bet
        return next_bet