"""
Dynamic Skip strategy implementation.
"""

import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class DynamicSkipStrategy(BaseStrategy):
    """
    Dynamic Skip strategy - intelligently skips betting in uncertain situations
    to minimize losses. This strategy focuses on identifying high-risk scenarios
    where the probability of a loss is elevated, and avoids betting in those situations.
    
    Key features:
    1. Dynamically identifies situations with historically poor performance
    2. Uses a tiered confidence system with multiple thresholds
    3. Builds a "risk profile" of different betting scenarios
    4. Progressively becomes more selective after losses
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate parameters
        self.base_confidence = self.params.get('base_confidence', 0.6)
        if not isinstance(self.base_confidence, (int, float)) or not 0 < self.base_confidence < 1:
            raise ValueError("base_confidence must be between 0 and 1")
            
        self.risk_threshold = self.params.get('risk_threshold', 0.4)
        if not isinstance(self.risk_threshold, (int, float)) or not 0 < self.risk_threshold < 1:
            raise ValueError("risk_threshold must be between 0 and 1")
            
        self.window_size = self.params.get('window_size', 15)
        if not isinstance(self.window_size, int) or self.window_size < 5:
            raise ValueError("window_size must be an integer >= 5")
            
        self.short_window = self.params.get('short_window', 5)
        if not isinstance(self.short_window, int) or self.short_window < 3:
            raise ValueError("short_window must be an integer >= 3")
            
        self.recovery_factor = self.params.get('recovery_factor', 0.05)
        if not isinstance(self.recovery_factor, (int, float)) or not 0 < self.recovery_factor < 1:
            raise ValueError("recovery_factor must be between 0 and 1")
        
        # Initialize state tracking
        self._dynamic_skip_state = {
            'current_threshold': self.base_confidence,
            'current_risk': self.risk_threshold,
            'performance_window': [],
            'risk_patterns': {},
            'last_pattern': None,
            'win_streak': 0,
            'loss_streak': 0,
            'total_bets': 0,
            'wins': 0,
            'last_prediction': None,
            'betting_patterns': []
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tBase confidence: {self.base_confidence}"
                   f"\n\tRisk threshold: {self.risk_threshold}"
                   f"\n\tWindow size: {self.window_size}"
                   f"\n\tShort window: {self.short_window}"
                   f"\n\tRecovery factor: {self.recovery_factor}")

    def get_bet(self, outcomes):
        """
        Determine next bet using dynamic skip strategy analysis
        """
        historical_outcomes = self._get_historical_outcomes(outcomes)
        
        if len(historical_outcomes) < self.short_window:
            return 'B'  # Default when not enough history
            
        state = self._dynamic_skip_state
        
        # Update state based on last bet result
        if state['last_prediction'] and len(historical_outcomes) > 0:
            last_outcome = historical_outcomes[-1]
            is_win = (state['last_prediction'] == last_outcome)
            
            # Update performance tracking
            state['performance_window'].append(1 if is_win else 0)
            if len(state['performance_window']) > 20:
                state['performance_window'].pop(0)
                
            # Update risk patterns
            if state['last_pattern'] and len(historical_outcomes) > 0:
                if state['last_pattern'] not in state['risk_patterns']:
                    state['risk_patterns'][state['last_pattern']] = {
                        'wins': 0, 'losses': 0, 'total': 0, 'risk': self.risk_threshold
                    }
                
                pattern_stats = state['risk_patterns'][state['last_pattern']]
                pattern_stats['total'] += 1
                if is_win:
                    pattern_stats['wins'] += 1
                    state['win_streak'] += 1
                    state['loss_streak'] = 0
                    state['wins'] += 1
                else:
                    pattern_stats['losses'] += 1
                    state['loss_streak'] += 1
                    state['win_streak'] = 0
                
                # Update risk level
                if pattern_stats['total'] >= 3:
                    win_rate = pattern_stats['wins'] / pattern_stats['total']
                    pattern_stats['risk'] = 1.0 - win_rate
            
            state['total_bets'] += 1
        
        # Extract current pattern features
        current_pattern_features = []
        
        # 1. Last n outcomes as a pattern
        pattern_length = min(4, len(historical_outcomes))
        if pattern_length > 0:
            current_pattern_features.append(f"last_{pattern_length}={''.join(historical_outcomes[-pattern_length:])}")
        
        # 2. Current streak type and length
        if len(historical_outcomes) > 0:
            streak_val = historical_outcomes[-1]
            streak_length = 1
            for i in range(len(historical_outcomes)-2, -1, -1):
                if historical_outcomes[i] == streak_val:
                    streak_length += 1
                else:
                    break
            current_pattern_features.append(f"streak_{streak_val}_{streak_length}")
            
        # 3. Short-term bias
        if len(historical_outcomes) >= self.short_window:
            short_outcomes = historical_outcomes[-self.short_window:]
            p_ratio = short_outcomes.count('P') / len(short_outcomes)
            bias_level = int(p_ratio * 10)
            current_pattern_features.append(f"bias_{bias_level}")
                
        # 4. Pattern structure (alternating vs runs)
        if len(historical_outcomes) >= 4:
            last_4 = historical_outcomes[-4:]
            alternating = True
            for i in range(1, len(last_4)):
                if last_4[i] == last_4[i-1]:
                    alternating = False
                    break
            current_pattern_features.append("alt" if alternating else "run")
            
        # Combine features into a pattern signature
        pattern_signature = "_".join(sorted(current_pattern_features))
        state['last_pattern'] = pattern_signature
        
        # Check risk level for this pattern
        pattern_risk = self._analyze_pattern_risk(pattern_signature, historical_outcomes, state)
            
        # Analyze outcomes to determine best bet
        p_confidence = 0.5
        b_confidence = 0.5
        
        # 1. Short-term frequency analysis
        if len(historical_outcomes) >= self.short_window:
            short_term = historical_outcomes[-self.short_window:]
            p_ratio = short_term.count('P') / len(short_term)
            
            # Bet against strong short-term bias
            if p_ratio > 0.7:
                b_confidence += 0.15
            elif p_ratio < 0.3:
                p_confidence += 0.15
                    
        # 2. Pattern-based prediction
        pattern_length = 4
        if len(historical_outcomes) >= pattern_length * 2:
            current = historical_outcomes[-pattern_length:]
            matches = []
            
            # Find similar patterns in history
            for i in range(len(historical_outcomes) - pattern_length * 2):
                past = historical_outcomes[i:i+pattern_length]
                if past == current and i + pattern_length < len(historical_outcomes):
                    matches.append(historical_outcomes[i + pattern_length])
            
            if matches:
                p_count = matches.count('P')
                match_ratio = p_count / len(matches)
                confidence_boost = min(0.2, 0.05 * len(matches))
                
                if match_ratio > 0.6:
                    p_confidence += confidence_boost
                elif match_ratio < 0.4:
                    b_confidence += confidence_boost
                    
        # 3. Streak analysis
        if len(historical_outcomes) >= 3:
            last = historical_outcomes[-1]
            streak = 1
            for i in range(len(historical_outcomes)-2, -1, -1):
                if historical_outcomes[i] == last:
                    streak += 1
                else:
                    break
            
            # Long streaks tend to break
            if streak >= 4:
                if last == 'P':
                    b_confidence += 0.1
                else:
                    p_confidence += 0.1
        
        # 4. Alternating pattern detection
        if len(historical_outcomes) >= 4:
            last_4 = historical_outcomes[-4:]
            if all(last_4[i] != last_4[i-1] for i in range(1, len(last_4))):
                next_predicted = 'P' if last_4[-1] == 'B' else 'B'
                if next_predicted == 'P':
                    p_confidence += 0.1
                else:
                    b_confidence += 0.1
        
        # Apply banker's advantage due to lower commission
        b_confidence *= 1.03  # 3% boost to reflect lower commission
        
        # Consider overall win rate of the strategy
        overall_win_rate = state['wins'] / state['total_bets'] if state['total_bets'] > 0 else 0.5
        
        # Adjust confidence threshold based on historical performance
        adaptive_threshold = state['current_threshold']
        
        # If overall strategy is performing poorly, be more selective
        if state['total_bets'] >= 10 and overall_win_rate < 0.45:
            adaptive_threshold += 0.05
        
        # Decision logic
        max_confidence = max(p_confidence, b_confidence)
        
        # Check if best confidence meets threshold
        if max_confidence < adaptive_threshold:
            return 'SKIP'
            
        # Check if pattern has high historical risk
        if pattern_risk > self.risk_threshold:
            return 'SKIP'
            
        # Proceed with the more confident bet
        if p_confidence > b_confidence:
            state['last_prediction'] = 'P'
            return 'P'
        else:
            state['last_prediction'] = 'B'
            return 'B'
    
    def _analyze_pattern_risk(self, pattern_signature, outcomes, state):
        """
        Analyze the risk level associated with a given pattern.
        
        Args:
            pattern_signature: The pattern identifier
            outcomes: Historical outcomes to analyze
            state: Strategy state dictionary
            
        Returns:
            float: Risk level between 0 and 1
        """
        if pattern_signature not in state['risk_patterns']:
            state['risk_patterns'][pattern_signature] = {
                'wins': 0,
                'losses': 0,
                'total': 0,
                'risk': self.risk_threshold
            }
            
        pattern_stats = state['risk_patterns'][pattern_signature]
        
        if pattern_stats['total'] >= 3:
            # Calculate win rate for this pattern
            win_rate = pattern_stats['wins'] / pattern_stats['total'] if pattern_stats['total'] > 0 else 0.5
            
            # Higher risk for patterns with poor performance
            if win_rate < 0.4:
                return min(0.8, self.risk_threshold + (0.4 - win_rate))
            # Lower risk for patterns with good performance    
            elif win_rate > 0.6:
                return max(0.2, self.risk_threshold - (win_rate - 0.6))
                
        return self.risk_threshold