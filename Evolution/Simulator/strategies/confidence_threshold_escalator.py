"""
Confidence Threshold Escalator strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ConfidenceThresholdEscalatorStrategy(BaseStrategy):
    """
    Confidence Threshold Escalator strategy - adjusts confidence requirements dynamically.
    
    This strategy always places a bet but escalates the confidence threshold required
    to stay on the same side after losses, encouraging side-switching in losing streaks.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.base_threshold = self.params.get('base_threshold', 0.55)
        self.escalation_factor = self.params.get('escalation_factor', 0.05)
        self.max_threshold = self.params.get('max_threshold', 0.85)
        self.de_escalation_factor = self.params.get('de_escalation_factor', 0.02)
        self.pattern_length = self.params.get('pattern_length', 4)
        self.window_sizes = self.params.get('window_sizes', [5, 10, 20])
        
        # Initialize state
        self._state = {
            'current_side': 'B',  # Start with banker due to lower house edge
            'current_threshold': self.base_threshold,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'last_confidence': {'P': 0.5, 'B': 0.5},
            'side_performance': {'P': {'wins': 0, 'losses': 0}, 'B': {'wins': 0, 'losses': 0}},
            'pattern_performance': {},
            'last_prediction': None
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using confidence threshold escalator approach
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.pattern_length:
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
                state['side_performance'][prediction]['wins'] += 1
                
                # De-escalate threshold after wins
                state['current_threshold'] = max(
                    self.base_threshold, 
                    state['current_threshold'] - self.de_escalation_factor
                )
                
                # Update pattern performance
                if len(historical_outcomes) >= self.pattern_length:
                    pattern = ''.join(historical_outcomes[-(self.pattern_length+1):-1])
                    if pattern not in state['pattern_performance']:
                        state['pattern_performance'][pattern] = {'P': 0, 'B': 0}
                    state['pattern_performance'][pattern][actual] += 1
            else:
                state['consecutive_losses'] += 1
                state['consecutive_wins'] = 0
                state['side_performance'][prediction]['losses'] += 1
                
                # Escalate threshold after losses
                state['current_threshold'] = min(
                    self.max_threshold,
                    state['current_threshold'] + self.escalation_factor
                )
        
        # Calculate confidence scores for each side
        p_confidence = 0.5  # Start at neutral
        b_confidence = 0.5  # Start at neutral
        
        # 1. Pattern-based confidence
        if len(historical_outcomes) >= self.pattern_length:
            pattern = ''.join(historical_outcomes[-self.pattern_length:])
            
            if pattern in state['pattern_performance']:
                stats = state['pattern_performance'][pattern]
                total = stats['P'] + stats['B']
                if total >= 3:  # Minimum observations
                    p_ratio = stats['P'] / total if total > 0 else 0.5
                    pattern_boost = 0.15  # Confidence boost from pattern matching
                    
                    if p_ratio > 0.6:
                        p_confidence += pattern_boost
                    elif p_ratio < 0.4:
                        b_confidence += pattern_boost
            
            # Also search for similar patterns in history
            matches = []
            for i in range(len(historical_outcomes) - self.pattern_length * 2):
                past = historical_outcomes[i:i+self.pattern_length]
                current = historical_outcomes[-self.pattern_length:]
                if past == current and i + self.pattern_length < len(historical_outcomes):
                    matches.append(historical_outcomes[i + self.pattern_length])
            
            if len(matches) >= 3:
                p_ratio = matches.count('P') / len(matches)
                match_boost = min(0.1, 0.02 * len(matches))
                
                if p_ratio > 0.6:
                    p_confidence += match_boost
                elif p_ratio < 0.4:
                    b_confidence += match_boost
        
        # 2. Streak-based confidence
        if len(historical_outcomes) >= 3:
            last = historical_outcomes[-1]
            streak = 1
            
            for i in range(len(historical_outcomes)-2, -1, -1):
                if historical_outcomes[i] == last:
                    streak += 1
                else:
                    break
            
            if streak >= 3:
                streak_boost = min(0.15, 0.05 * streak)
                # With long streaks, expect mean reversion
                if last == 'P':
                    b_confidence += streak_boost
                else:
                    p_confidence += streak_boost
        
        # 3. Multi-timeframe bias analysis
        for window in self.window_sizes:
            if len(historical_outcomes) >= window:
                window_outcomes = historical_outcomes[-window:]
                p_count = window_outcomes.count('P')
                b_count = window_outcomes.count('B')
                total = p_count + b_count
                
                if total > 0:
                    p_ratio = p_count / total
                    bias_boost = 0.05
                    
                    # Bet against strong bias (regression to mean)
                    if p_ratio > 0.65:
                        b_confidence += bias_boost
                    elif p_ratio < 0.35:
                        p_confidence += bias_boost
        
        # 4. Alternating pattern detection
        if len(historical_outcomes) >= 4:
            last_4 = historical_outcomes[-4:]
            alternating = True
            for i in range(1, len(last_4)):
                if last_4[i] == last_4[i-1]:
                    alternating = False
                    break
            
            if alternating:
                alt_boost = 0.1
                # If alternating pattern, continue it
                if last_4[-1] == 'P':
                    b_confidence += alt_boost
                else:
                    p_confidence += alt_boost
        
        # 5. Historical side performance
        for side in ['P', 'B']:
            stats = state['side_performance'][side]
            total = stats['wins'] + stats['losses']
            
            if total >= 5:  # Minimum observations
                win_rate = stats['wins'] / total if total > 0 else 0.5
                perf_boost = 0.05
                
                if side == 'P' and win_rate > 0.55:
                    p_confidence += perf_boost
                elif side == 'B' and win_rate > 0.55:
                    b_confidence += perf_boost
        
        # Apply banker edge (due to lower commission)
        b_confidence *= 1.02  # 2% advantage for banker bets
        
        # Store confidences for next round
        state['last_confidence']['P'] = p_confidence
        state['last_confidence']['B'] = b_confidence
        
        # Decision logic - compare confidence to current threshold
        current_threshold = state['current_threshold']
        logger.debug(f"CONFIDENCE_ESCALATOR: Confidence scores - P: {p_confidence:.2f}, B: {b_confidence:.2f}, threshold: {current_threshold:.2f}")
        
        # If current side's confidence exceeds threshold, stay with it
        if state['current_side'] == 'P' and p_confidence >= b_confidence and p_confidence >= current_threshold:
            next_bet = 'P'
        elif state['current_side'] == 'B' and b_confidence >= p_confidence and b_confidence >= current_threshold:
            next_bet = 'B'
        elif p_confidence > b_confidence:
            next_bet = 'P'
            state['current_side'] = 'P'
            state['current_threshold'] = self.base_threshold  # Reset threshold on side switch
        else:
            next_bet = 'B'
            state['current_side'] = 'B'
            state['current_threshold'] = self.base_threshold  # Reset threshold on side switch
        
        # Store prediction for next round
        state['last_prediction'] = next_bet
        
        # Always return a legitimate bet (never SKIP)
        return next_bet