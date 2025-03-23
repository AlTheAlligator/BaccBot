"""
Loss Aversion strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class LossAversionStrategy(BaseStrategy):
    """
    Loss Aversion strategy - designed to reduce consecutive losses by adapting 
    behavior after losses and being more conservative when there's uncertainty.
    
    Key features:
    1. Dynamically adjusts risk tolerance based on recent performance
    2. Uses banker bias (lower commission) after losses
    3. Increases threshold for betting after consecutive losses
    4. Combines multiple signals with variable weightings based on win/loss history
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.base_confidence = self.params.get('base_confidence', 0.55)
        self.recovery_threshold = self.params.get('recovery_threshold', 2)
        self.window_size = self.params.get('window_size', 15)
        self.short_window = self.params.get('short_window', 5)
        self.banking_emphasis = self.params.get('banking_emphasis', 0.0)
        self.recovery_intensity = self.params.get('recovery_intensity', 1.0)
        self.max_skip_count = self.params.get('max_skip_count', 0)
        self.pattern_boost = self.params.get('pattern_boost', 0.0)
        self.adaptive_mode = self.params.get('adaptive_mode', False)
        
        # Initialize state
        self._loss_aversion_state = {
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'recovery_mode': False,
            'last_prediction': None,
            'skipped_count': 0,
            'signal_success': {},
            'last_decisions': [],
            'last_outcomes': [],
            'win_history': []
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using loss aversion strategy
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.short_window:
            return 'B'  # Default when not enough history
        
        state = self._loss_aversion_state
        
        # Update state based on last bet result
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            # Update tracking data
            state['win_history'].append(1 if is_win else 0)
            if len(state['win_history']) > 20:  # Keep last 20 results
                state['win_history'].pop(0)
                
            # Update consecutive win/loss tracking
            if is_win:
                state['consecutive_wins'] += 1
                state['consecutive_losses'] = 0
                
                # Update signal success tracking
                for signal in state.get('last_signals', []):
                    if signal not in state['signal_success']:
                        state['signal_success'][signal] = {'wins': 0, 'losses': 0}
                    state['signal_success'][signal]['wins'] += 1
            else:
                state['consecutive_losses'] += 1
                state['consecutive_wins'] = 0
                
                # Update signal success tracking
                for signal in state.get('last_signals', []):
                    if signal not in state['signal_success']:
                        state['signal_success'][signal] = {'wins': 0, 'losses': 0}
                    state['signal_success'][signal]['losses'] += 1
                
            # Record the decision and outcome
            state['last_decisions'].append(prediction)
            state['last_outcomes'].append(actual)
            if len(state['last_decisions']) > 10:
                state['last_decisions'].pop(0)
                state['last_outcomes'].pop(0)
                
        # Reset skipped count if we're not in recovery mode
        if not state['recovery_mode']:
            state['skipped_count'] = 0
                
        # Determine if we're in recovery mode after consecutive losses
        state['recovery_mode'] = state['consecutive_losses'] >= self.recovery_threshold
        
        # Adjust confidence threshold based on win/loss history and recovery mode
        confidence_threshold = self.base_confidence
        if state['recovery_mode']:
            # Increase required confidence when in recovery mode
            recovery_boost = state['consecutive_losses'] * 0.03 * self.recovery_intensity
            confidence_threshold += min(0.2, recovery_boost)
        
        # Adapt threshold based on overall performance if adaptive mode is on
        if self.adaptive_mode and len(state['win_history']) >= 10:
            win_rate = sum(state['win_history']) / len(state['win_history'])
            if win_rate < 0.4:
                confidence_threshold += 0.05  # Become more selective when doing poorly
            elif win_rate > 0.6:
                confidence_threshold -= 0.03  # Be more aggressive when doing well
        
        # 1. Recent trend analysis - looking for patterns in most recent outcomes
        recent_outcomes = historical_outcomes[-self.short_window:]
        
        # Check for consistent alternating pattern (high confidence signal)
        alternating_pattern = True
        for i in range(1, len(recent_outcomes)):
            if recent_outcomes[i] == recent_outcomes[i-1]:
                alternating_pattern = False
                break
                
        alternating_confidence = 0.0
        alternating_prediction = None
        
        if alternating_pattern and len(recent_outcomes) >= 4:
            alternating_confidence = 0.7  # High confidence for clean alternating pattern
            alternating_prediction = 'P' if recent_outcomes[-1] == 'B' else 'B'
            
        # 2. Bias detection in a sliding window
        window_outcomes = historical_outcomes[-self.window_size:] if len(historical_outcomes) >= self.window_size else historical_outcomes
        p_count = window_outcomes.count('P')
        b_count = window_outcomes.count('B')
        
        bias_confidence = 0.0
        bias_prediction = None
        
        if len(window_outcomes) > 0:
            total = p_count + b_count
            p_ratio = p_count / total if total > 0 else 0.5
            
            if p_ratio > 0.65:
                bias_confidence = 0.6 + min(0.15, (p_ratio - 0.65) * 2)  # More confidence for stronger bias
                bias_prediction = 'B'  # Bet against strong Player bias
            elif p_ratio < 0.35:
                bias_confidence = 0.6 + min(0.15, (0.35 - p_ratio) * 2)
                bias_prediction = 'P'  # Bet against strong Banker bias
                
        # 3. Streak analysis
        streak_length = 1
        streak_value = historical_outcomes[-1] if historical_outcomes else None
        for i in range(len(historical_outcomes)-2, -1, -1):
            if historical_outcomes[i] == streak_value:
                streak_length += 1
            else:
                break
                
        streak_confidence = 0.0
        streak_prediction = None
        
        if streak_length >= 3:
            streak_confidence = 0.6 + min(0.2, (streak_length - 3) * 0.05)  # More confidence for longer streaks
            streak_prediction = 'P' if streak_value == 'B' else 'B'  # Bet against streak
                
        # 4. Pattern matching with historical data
        pattern_length = 4
        pattern_confidence = 0.0
        pattern_prediction = None
        
        if len(historical_outcomes) >= pattern_length * 2:
            current_pattern = historical_outcomes[-pattern_length:]
            pattern_matches = []
            
            for i in range(len(historical_outcomes) - pattern_length * 2):
                past_pattern = historical_outcomes[i:i+pattern_length]
                if past_pattern == current_pattern and i + pattern_length < len(historical_outcomes):
                    pattern_matches.append(historical_outcomes[i + pattern_length])
            
            if len(pattern_matches) >= 3:
                p_ratio = pattern_matches.count('P') / len(pattern_matches)
                pattern_boost = min(0.2, 0.05 * len(pattern_matches) + self.pattern_boost)
                
                if p_ratio > 0.6:
                    pattern_confidence = 0.6 + pattern_boost
                    pattern_prediction = 'P'
                elif p_ratio < 0.4:
                    pattern_confidence = 0.6 + pattern_boost
                    pattern_prediction = 'B'
                
        # 5. Analysis of skipped opportunities (if in recovery mode)
        # Look at what outcomes occurred when we weren't betting
        skip_analysis_confidence = 0.0
        skip_prediction = None
        
        if state['recovery_mode'] and len(state['last_decisions']) >= 3:
            skipped_outcomes = []
            for i, decision in enumerate(state['last_decisions']):
                if decision == 'SKIP' and i < len(state['last_outcomes']):
                    skipped_outcomes.append(state['last_outcomes'][i])
                    
            if skipped_outcomes:
                p_ratio = skipped_outcomes.count('P') / len(skipped_outcomes)
                if p_ratio > 0.7:
                    skip_analysis_confidence = 0.6
                    skip_prediction = 'P'  # We missed mostly P, bet on P
                elif p_ratio < 0.3:
                    skip_analysis_confidence = 0.6
                    skip_prediction = 'B'  # We missed mostly B, bet on B
                                    
        # 6. Banker bias in recovery mode (safer option due to lower house edge)
        banker_confidence = 0.0
        banker_prediction = 'B'
        
        if state['recovery_mode'] and state['consecutive_losses'] >= 3:
            banker_confidence = 0.55 + min(0.15, state['consecutive_losses'] * 0.02) + self.banking_emphasis
            
        # Combine all signals - prioritize higher confidence
        confidence_signals = []
        state['last_signals'] = []
        
        if alternating_confidence > confidence_threshold:
            confidence_signals.append((alternating_confidence, alternating_prediction, 'alternating'))
            state['last_signals'].append('alternating')
            
        if bias_confidence > confidence_threshold:
            confidence_signals.append((bias_confidence, bias_prediction, 'bias'))
            state['last_signals'].append('bias')
            
        if streak_confidence > confidence_threshold:
            confidence_signals.append((streak_confidence, streak_prediction, 'streak'))
            state['last_signals'].append('streak')
            
        if pattern_confidence > confidence_threshold:
            confidence_signals.append((pattern_confidence, pattern_prediction, 'pattern'))
            state['last_signals'].append('pattern')
            
        if skip_analysis_confidence > confidence_threshold:
            confidence_signals.append((skip_analysis_confidence, skip_prediction, 'skip_analysis'))
            state['last_signals'].append('skip_analysis')
            
        if banker_confidence > confidence_threshold:
            confidence_signals.append((banker_confidence, banker_prediction, 'banker_bias'))
            state['last_signals'].append('banker_bias')
            
        # Make decision based on signals
        if not confidence_signals:
            if state['recovery_mode'] and self.max_skip_count > 0 and state['skipped_count'] < self.max_skip_count:
                # Skip betting if no strong signals during recovery mode
                state['skipped_count'] += 1
                state['last_prediction'] = 'SKIP'
                return 'SKIP'
            else:
                # Default to banker (lower house edge) if no strong signals
                state['last_prediction'] = 'B'
                return 'B'
                
        # Sort by confidence (descending)
        confidence_signals.sort(key=lambda x: x[0], reverse=True)
        
        # If we're in recovery mode and the signals aren't very strong, skip betting
        if state['recovery_mode'] and confidence_signals[0][0] < 0.65 and self.max_skip_count > 0 and state['skipped_count'] < self.max_skip_count:
            state['skipped_count'] += 1
            state['last_prediction'] = 'SKIP'
            return 'SKIP'
            
        # Go with highest confidence prediction
        best_prediction = confidence_signals[0][1]
        state['last_prediction'] = best_prediction
        return best_prediction