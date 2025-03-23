"""
Risk Parity strategy implementation.
"""

import logging
import numpy as np
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RiskParityStrategy(BaseStrategy):
    """
    Risk Parity strategy - focuses on managing risk exposure by dynamically
    adjusting bet frequency based on performance and volatility.
    
    This strategy uses concepts from portfolio management to balance risk
    across different betting approaches.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.lookback_window = self.params.get('lookback_window', 30)
        self.min_confidence = self.params.get('min_confidence', 0.6)
        self.max_volatility = self.params.get('max_volatility', 0.5)
        self.skip_ratio = self.params.get('skip_ratio', 0.3)
        self.risk_limit = self.params.get('risk_limit', 3.0)
        
        # Initialize state
        self._state = {
            'last_prediction': None,
            'performance': [],
            'win_streak': 0,
            'loss_streak': 0,
            'current_volatility': 0.2,  # Starting volatility estimate
            'signal_stats': {},
            'last_signals': []
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using risk parity approach
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < 10:
            return 'B'  # Default when not enough history
        
        state = self._state
        
        # Update performance tracking
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            # Update overall performance
            state['performance'].append(1 if is_win else 0)
            if len(state['performance']) > self.lookback_window:
                state['performance'].pop(0)
            
            # Update win/loss streaks
            if is_win:
                state['win_streak'] += 1
                state['loss_streak'] = 0
            else:
                state['loss_streak'] += 1
                state['win_streak'] = 0
                
            # Update signal stats
            for signal in state['last_signals']:
                if signal not in state['signal_stats']:
                    state['signal_stats'][signal] = {'wins': 0, 'losses': 0, 'total': 0}
                
                state['signal_stats'][signal]['total'] += 1
                if is_win:
                    state['signal_stats'][signal]['wins'] += 1
                else:
                    state['signal_stats'][signal]['losses'] += 1
        
        # Calculate current volatility from recent performance
        if len(state['performance']) >= 10:
            # Use rolling window standard deviation as volatility measure
            state['current_volatility'] = np.std(state['performance'][-10:])
        
        # Generate signals from different approaches
        signals = {}
        
        # 1. Pattern matching signal
        pattern_signal, pattern_conf = self._generate_pattern_signal(historical_outcomes)
        if pattern_conf >= self.min_confidence:
            signals['pattern'] = pattern_signal
            
        # 2. Streak analysis signal
        streak_signal, streak_conf = self._generate_streak_signal(historical_outcomes)
        if streak_conf >= self.min_confidence:
            signals['streak'] = streak_signal
            
        # 3. Frequency analysis signal
        freq_signal, freq_conf = self._generate_frequency_signal(historical_outcomes)
        if freq_conf >= self.min_confidence:
            signals['frequency'] = freq_signal
            
        # 4. Alternating pattern signal
        alt_signal, alt_conf = self._generate_alternating_signal(historical_outcomes)
        if alt_conf >= self.min_confidence:
            signals['alternating'] = alt_signal
        
        # Adjust signal weights based on historical performance
        signal_weights = {}
        for signal_name in signals.keys():
            if signal_name in state['signal_stats']:
                stats = state['signal_stats'][signal_name]
                if stats['total'] >= 5:
                    win_rate = stats['wins'] / stats['total']
                    # Higher weights for better performing signals
                    signal_weights[signal_name] = max(0.1, min(2.0, win_rate * 2))
                else:
                    signal_weights[signal_name] = 1.0  # Default weight
            else:
                signal_weights[signal_name] = 1.0  # Default weight
        
        # Calculate vote tallies with weights
        p_votes = sum(signal_weights[s] for s, v in signals.items() if v == 'P')
        b_votes = sum(signal_weights[s] for s, v in signals.items() if v == 'B')
        
        # Calculate risk exposure based on volatility and loss streak
        risk_exposure = state['current_volatility'] * (1 + 0.25 * state['loss_streak'])
        
        # Store signals for next iteration
        state['last_signals'] = list(signals.keys())
        
        # Decision logic considering risk exposure
        if risk_exposure > self.max_volatility:
            # High risk environment - be selective
            if max(p_votes, b_votes) < self.risk_limit:
                return 'SKIP'  # Not enough strong signals in high risk environment
            elif p_votes > b_votes * 1.5:  # Much stronger P signal
                next_bet = 'P'
            elif b_votes > p_votes * 1.5:  # Much stronger B signal
                next_bet = 'B'
            else:
                return 'SKIP'  # Uncertain in high risk environment
        else:
            # Normal risk environment
            skip_threshold = self.skip_ratio if state['loss_streak'] <= 1 else self.skip_ratio * (1 + 0.1 * state['loss_streak'])
            
            if np.random.random() < skip_threshold and p_votes == 0 and b_votes == 0:
                # Randomly skip some percentage of bets with no signals
                return 'SKIP'
                
            # Decision based on weighted votes
            if p_votes > b_votes:
                next_bet = 'P'
            elif b_votes > p_votes:
                next_bet = 'B'
            else:
                # Apply banker bias in case of tie
                next_bet = 'B'
        
        # Store prediction
        state['last_prediction'] = next_bet
        return next_bet
    
    def _generate_pattern_signal(self, outcomes):
        """Generate signal based on pattern matching"""
        pattern_length = 4
        confidence = 0.5
        
        if len(outcomes) < pattern_length * 2:
            return None, confidence
            
        current = outcomes[-pattern_length:]
        matches = []
        
        for i in range(len(outcomes) - pattern_length * 2):
            past = outcomes[i:i+pattern_length]
            if past == current and i + pattern_length < len(outcomes):
                matches.append(outcomes[i + pattern_length])
        
        if len(matches) >= 3:
            p_count = matches.count('P')
            total = len(matches)
            p_ratio = p_count / total
            
            if p_ratio >= 0.67:
                confidence = 0.5 + min(0.4, 0.05 * len(matches))
                return 'P', confidence
            elif p_ratio <= 0.33:
                confidence = 0.5 + min(0.4, 0.05 * len(matches))
                return 'B', confidence
                
        return None, confidence
    
    def _generate_streak_signal(self, outcomes):
        """Generate signal based on streak analysis"""
        if len(outcomes) < 3:
            return None, 0.5
            
        last = outcomes[-1]
        streak = 1
        for i in range(len(outcomes)-2, -1, -1):
            if outcomes[i] == last:
                streak += 1
            else:
                break
                
        if streak >= 3:
            # Long streaks tend to reverse - stronger signal for longer streaks
            confidence = 0.5 + min(0.35, 0.05 * streak)
            return 'P' if last == 'B' else 'B', confidence
            
        return None, 0.5
    
    def _generate_frequency_signal(self, outcomes):
        """Generate signal based on frequency analysis"""
        if len(outcomes) < 10:
            return None, 0.5
            
        # Check multiple window sizes for bias
        windows = [10, 20] if len(outcomes) >= 20 else [10]
        max_confidence = 0.5
        best_signal = None
        
        for window in windows:
            recent = outcomes[-window:]
            p_count = recent.count('P')
            b_count = recent.count('B')
            total = p_count + b_count
            
            if total > 0:
                p_ratio = p_count / total
                
                if p_ratio >= 0.7:  # Strong Player bias
                    confidence = 0.5 + min(0.3, (p_ratio - 0.5) * 2)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_signal = 'B'  # Bet against strong bias
                elif p_ratio <= 0.3:  # Strong Banker bias
                    confidence = 0.5 + min(0.3, (0.5 - p_ratio) * 2)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_signal = 'P'  # Bet against strong bias
                        
        return best_signal, max_confidence
    
    def _generate_alternating_signal(self, outcomes):
        """Generate signal based on alternating pattern detection"""
        if len(outcomes) < 4:
            return None, 0.5
            
        # Check for alternating pattern in last 4 outcomes
        last_4 = outcomes[-4:]
        alternating = True
        
        for i in range(1, len(last_4)):
            if last_4[i] == last_4[i-1]:
                alternating = False
                break
                
        if alternating:
            # If alternating, predict the next in sequence with high confidence
            confidence = 0.75  # Strong confidence for alternating patterns
            prediction = 'P' if last_4[-1] == 'B' else 'B'
            return prediction, confidence
            
        return None, 0.5