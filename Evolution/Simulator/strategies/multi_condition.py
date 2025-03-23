"""
Multi-Condition strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MultiConditionStrategy(BaseStrategy):
    """
    Multi-Condition strategy - requires multiple conditions to be satisfied
    before making a bet.
    
    This strategy uses logical AND operations to combine multiple signals and
    only places bets when all conditions are satisfied.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        self.window_size = self.params.get('window_size', 15)
        self.short_window = self.params.get('short_window', 5)
        self.pattern_length = self.params.get('pattern_length', 4)
        self.streak_threshold = self.params.get('streak_threshold', 3)
        self.skip_enabled = self.params.get('skip_enabled', True)
        self.conditions_required = self.params.get('conditions_required', 2)
        
        # Initialize state
        self._state = {
            'last_prediction': None,
            'performance': [],
            'condition_performance': {
                'streak': [],
                'pattern': [],
                'bias': [],
                'alternating': []
            }
        }
    
    def get_bet(self, outcomes):
        """
        Determine next bet using multiple condition checks
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < max(self.window_size, self.pattern_length):
            return 'B'  # Default when not enough history
        
        state = self._state
        
        # Update performance tracking
        if state['last_prediction'] and len(outcomes) > 1:
            actual = outcomes[-2]  # Previous outcome
            prediction = state['last_prediction']
            is_win = (prediction == actual)
            
            # Update overall performance
            state['performance'].append(1 if is_win else 0)
            if len(state['performance']) > 20:  # Keep last 20 results
                state['performance'].pop(0)
            
            # Update condition performance
            for condition, result_list in state.get('last_condition_results', {}).items():
                if result_list[0]:  # If condition was satisfied
                    if condition not in state['condition_performance']:
                        state['condition_performance'][condition] = []
                    
                    state['condition_performance'][condition].append(1 if is_win else 0)
                    if len(state['condition_performance'][condition]) > 20:
                        state['condition_performance'][condition].pop(0)
        
        # Initialize condition results
        condition_results = {
            'streak': [False, None],
            'pattern': [False, None],
            'bias': [False, None],
            'alternating': [False, None]
        }
        
        # 1. Streak condition
        if len(historical_outcomes) >= self.streak_threshold:
            last_value = historical_outcomes[-1]
            streak_length = 1
            for i in range(len(historical_outcomes)-2, -1, -1):
                if historical_outcomes[i] == last_value:
                    streak_length += 1
                else:
                    break
            
            if streak_length >= self.streak_threshold:
                opposite = 'P' if last_value == 'B' else 'B'
                condition_results['streak'] = [True, opposite]
        
        # 2. Pattern matching condition
        if len(historical_outcomes) >= self.pattern_length * 2:
            current_pattern = historical_outcomes[-self.pattern_length:]
            matches = []
            
            # Find similar patterns in history
            for i in range(len(historical_outcomes) - self.pattern_length * 2):
                past_pattern = historical_outcomes[i:i+self.pattern_length]
                if past_pattern == current_pattern and i + self.pattern_length < len(historical_outcomes):
                    matches.append(historical_outcomes[i + self.pattern_length])
            
            if len(matches) >= 3:
                p_count = matches.count('P')
                b_count = matches.count('B')
                total = p_count + b_count
                
                if total > 0:
                    p_ratio = p_count / total
                    if p_ratio >= 0.67:
                        condition_results['pattern'] = [True, 'P']
                    elif p_ratio <= 0.33:
                        condition_results['pattern'] = [True, 'B']
        
        # 3. Bias condition
        short_outcomes = historical_outcomes[-self.short_window:]
        p_count = short_outcomes.count('P')
        b_count = short_outcomes.count('B')
        total = p_count + b_count
        
        if total > 0:
            p_ratio = p_count / total
            if p_ratio >= 0.8:  # Strong Player bias
                condition_results['bias'] = [True, 'B']  # Bet against bias
            elif p_ratio <= 0.2:  # Strong Banker bias
                condition_results['bias'] = [True, 'P']  # Bet against bias
                
        # 4. Alternating pattern condition
        if len(historical_outcomes) >= 4:
            last_4 = historical_outcomes[-4:]
            alternating = True
            for i in range(1, len(last_4)):
                if last_4[i] == last_4[i-1]:
                    alternating = False
                    break
            
            if alternating:
                next_in_sequence = 'P' if last_4[-1] == 'B' else 'B'
                condition_results['alternating'] = [True, next_in_sequence]
        
        # Calculate condition weights based on historical performance
        condition_weights = {}
        for condition, perf_list in state['condition_performance'].items():
            if len(perf_list) >= 5:
                win_rate = sum(perf_list) / len(perf_list)
                # Conditions with better historical performance get higher weight
                condition_weights[condition] = win_rate
            else:
                condition_weights[condition] = 0.5  # Default when not enough data
        
        # Count satisfied conditions for each side
        weighted_votes = {'P': 0, 'B': 0}
        raw_votes = {'P': 0, 'B': 0}
        
        for condition, (satisfied, prediction) in condition_results.items():
            if satisfied and prediction in ['P', 'B']:
                weight = condition_weights.get(condition, 0.5)
                weighted_votes[prediction] += weight
                raw_votes[prediction] += 1
        
        # Store condition results for next round
        state['last_condition_results'] = condition_results
        
        # Decision logic - require minimum number of conditions
        total_conditions_satisfied = sum(1 for result in condition_results.values() if result[0])
        
        if total_conditions_satisfied >= self.conditions_required:
            # Use either weighted voting or majority voting
            if weighted_votes['P'] > weighted_votes['B']:
                next_bet = 'P'
            elif weighted_votes['P'] < weighted_votes['B']:
                next_bet = 'B'
            else:
                # If weighted votes are tied, use raw count
                if raw_votes['P'] > raw_votes['B']:
                    next_bet = 'P'
                elif raw_votes['P'] < raw_votes['B']:
                    next_bet = 'B'
                else:
                    next_bet = 'B'  # Default to banker if tied
        else:
            # Not enough conditions satisfied
            if self.skip_enabled:
                next_bet = 'SKIP'
            else:
                next_bet = 'B'  # Default to banker due to lower commission
        
        # Store prediction
        state['last_prediction'] = next_bet
        return next_bet