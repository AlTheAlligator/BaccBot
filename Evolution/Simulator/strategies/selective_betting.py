"""
Selective Betting strategy implementation.
"""

import logging
import numpy as np
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SelectiveBettingStrategy(BaseStrategy):
    """
    Selective Betting strategy - only bets on highest confidence opportunities.
    Focuses on quality over quantity by using multiple confidence thresholds and
    only betting when there's a strong statistical advantage.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.min_confidence = self.params.get('min_confidence', 0.7)
        self.ultra_confidence = self.params.get('ultra_confidence', 0.85)
        self.pattern_length = self.params.get('pattern_length', 4)
        self.window_sizes = self.params.get('window_sizes', [5, 10, 20])
        self.losing_streak_threshold = self.params.get('losing_streak_threshold', 2)
        
        # Initialize state
        self._state = {
            'loss_streak': 0,
            'win_streak': 0,
            'current_threshold': self.min_confidence,
            'last_prediction': None,
            'pattern_history': {},
            'results_history': []
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using selective betting strategy
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < max(self.window_sizes):
            return 'SKIP'  # Skip when not enough history
            
        state = self._state
        
        # Update performance tracking
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            result = 1 if prediction == actual else 0
            state['results_history'].append(result)
            
            if len(state['results_history']) > 30:
                state['results_history'].pop(0)
                
            if result == 1:  # Win
                state['win_streak'] += 1
                state['loss_streak'] = 0
            else:  # Loss
                state['loss_streak'] += 1
                state['win_streak'] = 0
                
        # Dynamically adjust confidence threshold based on recent performance
        if state['loss_streak'] > self.losing_streak_threshold:
            # Become more selective after losses
            state['current_threshold'] = min(self.min_confidence + (state['loss_streak'] - self.losing_streak_threshold) * 0.03, 
                                         self.ultra_confidence)
        else:
            # Reset to base threshold after good performance
            state['current_threshold'] = self.min_confidence
        
        # Calculate multiple confidence scores using different methods
        confidence_scores = {}
        
        # 1. Multi-timeframe bias analysis
        tf_confidence = self._multi_timeframe_analysis(historical_outcomes)
        confidence_scores['timeframe'] = tf_confidence
        
        # 2. Pattern matching confidence
        pattern_confidence = self._pattern_analysis(historical_outcomes)
        confidence_scores['pattern'] = pattern_confidence
        
        # 3. Streak analysis
        streak_confidence = self._streak_analysis(historical_outcomes)
        confidence_scores['streak'] = streak_confidence
        
        # 4. Frequency analysis with mean reversion
        freq_confidence = self._frequency_analysis(historical_outcomes)
        confidence_scores['frequency'] = freq_confidence
        
        # 5. Alternating pattern analysis
        alt_confidence = self._alternating_analysis(historical_outcomes)
        confidence_scores['alternating'] = alt_confidence
        
        # Combine all confidence scores
        # Weighted average based on historical reliability of each method
        weights = {'timeframe': 0.3, 'pattern': 0.25, 'streak': 0.2, 
                  'frequency': 0.15, 'alternating': 0.1}
        
        p_scores = {k: v.get('P', 0.5) for k, v in confidence_scores.items()}
        b_scores = {k: v.get('B', 0.5) for k, v in confidence_scores.items()}
        
        # Calculate weighted confidence
        p_confidence = sum(score * weights[method] for method, score in p_scores.items())
        b_confidence = sum(score * weights[method] for method, score in b_scores.items())
        
        # Apply slight banker bias due to lower commission
        b_confidence *= 1.02
        
        # Determine best bet and confidence level
        if p_confidence > b_confidence:
            best_bet = 'P'
            confidence = p_confidence
        else:
            best_bet = 'B'
            confidence = b_confidence
        
        # Only bet if confidence exceeds threshold
        if confidence >= state['current_threshold']:
            state['last_prediction'] = best_bet
            return best_bet
        else:
            return 'SKIP'
    
    def _multi_timeframe_analysis(self, outcomes):
        """Analyze bias across multiple timeframes"""
        confidence = {'P': 0.5, 'B': 0.5}
        
        for window in self.window_sizes:
            if len(outcomes) >= window:
                window_outcomes = outcomes[-window:]
                p_ratio = window_outcomes.count('P') / len(window_outcomes)
                
                # Strong bias in either direction
                if p_ratio > 0.65:
                    confidence['B'] += 0.1 * (2 - window/max(self.window_sizes))
                elif p_ratio < 0.35:
                    confidence['P'] += 0.1 * (2 - window/max(self.window_sizes))
        
        return confidence
    
    def _pattern_analysis(self, outcomes):
        """Analyze patterns in recent outcomes"""
        confidence = {'P': 0.5, 'B': 0.5}
        state = self._state
        
        if len(outcomes) < self.pattern_length * 2:
            return confidence
            
        # Current pattern
        current = ''.join(outcomes[-self.pattern_length:])
        
        # Update pattern history if we have a previous prediction
        if state['last_prediction'] and len(outcomes) > self.pattern_length:
            prev_pattern = ''.join(outcomes[-self.pattern_length-1:-1])
            actual = outcomes[-1]
            
            if prev_pattern not in state['pattern_history']:
                state['pattern_history'][prev_pattern] = {'P': 0, 'B': 0, 'total': 0}
                
            state['pattern_history'][prev_pattern][actual] += 1
            state['pattern_history'][prev_pattern]['total'] += 1
        
        # Search for this pattern in history
        if current in state['pattern_history'] and state['pattern_history'][current]['total'] >= 5:
            stats = state['pattern_history'][current]
            p_prob = stats['P'] / stats['total']
            
            if p_prob > 0.6:
                confidence['P'] += 0.2
            elif p_prob < 0.4:
                confidence['B'] += 0.2
        
        # Also search historical data for similar patterns
        matches = []
        for i in range(len(outcomes) - self.pattern_length * 2):
            past = ''.join(outcomes[i:i+self.pattern_length])
            if past == current and i + self.pattern_length < len(outcomes):
                matches.append(outcomes[i + self.pattern_length])
        
        if len(matches) >= 3:
            p_ratio = matches.count('P') / len(matches)
            
            if p_ratio > 0.6:
                confidence['P'] += min(0.2, 0.05 * len(matches))
            elif p_ratio < 0.4:
                confidence['B'] += min(0.2, 0.05 * len(matches))
                
        return confidence
    
    def _streak_analysis(self, outcomes):
        """Analyze streaks in the outcomes"""
        confidence = {'P': 0.5, 'B': 0.5}
        
        if len(outcomes) < 3:
            return confidence
            
        # Find current streak
        last = outcomes[-1]
        streak = 1
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == last:
                streak += 1
            else:
                break
                
        # Mean reversion theory - long streaks tend to break
        if streak >= 4:
            if last == 'P':
                confidence['B'] += 0.1 * min(streak/5, 2)
            else:
                confidence['P'] += 0.1 * min(streak/5, 2)
        
        return confidence
    
    def _frequency_analysis(self, outcomes):
        """Analyze frequency distribution with mean reversion"""
        confidence = {'P': 0.5, 'B': 0.5}
        
        for window in self.window_sizes:
            if len(outcomes) >= window:
                window_outcomes = outcomes[-window:]
                p_count = window_outcomes.count('P')
                b_count = window_outcomes.count('B')
                
                p_ratio = p_count / (p_count + b_count) if (p_count + b_count) > 0 else 0.5
                
                # Strong imbalance suggests mean reversion
                if window <= 10:  # Only apply to shorter windows
                    if p_ratio > 0.7:
                        confidence['B'] += 0.1
                    elif p_ratio < 0.3:
                        confidence['P'] += 0.1
        
        return confidence
    
    def _alternating_analysis(self, outcomes):
        """Analyze alternating patterns"""
        confidence = {'P': 0.5, 'B': 0.5}
        
        if len(outcomes) < 4:
            return confidence
            
        # Check if we have alternating pattern in last 4 outcomes
        last_4 = outcomes[-4:]
        alternating = True
        for i in range(1, len(last_4)):
            if last_4[i] == last_4[i-1]:
                alternating = False
                break
                
        if alternating:
            # If we have alternating pattern, continue it
            next_expected = 'P' if last_4[-1] == 'B' else 'B'
            confidence[next_expected] += 0.2
            
        return confidence