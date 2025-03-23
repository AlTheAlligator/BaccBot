"""
Dynamic Adaptive strategy implementation.
"""

import logging
import numpy as np
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class DynamicAdaptiveStrategy(BaseStrategy):
    """
    Dynamic Adaptive strategy - uses reinforcement learning concepts to adjust
    its approach based on performance feedback.
    
    This strategy continuously evaluates which betting approaches are working
    and dynamically shifts its behavior to maximize success rates.
    """
    
    def __init__(self, simulator, params=None):
        super().__init__(simulator, params)
        
        # Validate parameters
        self.learning_rate = self.params.get('learning_rate', 0.1)
        if not isinstance(self.learning_rate, (int, float)) or not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
            
        self.exploration_rate = self.params.get('exploration_rate', 0.2)
        if not isinstance(self.exploration_rate, (int, float)) or not 0 < self.exploration_rate < 1:
            raise ValueError("exploration_rate must be between 0 and 1")
            
        self.pattern_length = self.params.get('pattern_length', 4)
        if not isinstance(self.pattern_length, int) or self.pattern_length < 2:
            raise ValueError("pattern_length must be an integer >= 2")
            
        self.min_threshold = self.params.get('min_threshold', 0.55)
        if not isinstance(self.min_threshold, (int, float)) or not 0 < self.min_threshold < 1:
            raise ValueError("min_threshold must be between 0 and 1")
            
        self.action_history_size = self.params.get('action_history_size', 30)
        if not isinstance(self.action_history_size, int) or self.action_history_size < 5:
            raise ValueError("action_history_size must be an integer >= 5")

        # Initialize the adaptive state with defaulted action weights
        self._state = {
            'action_weights': {
                'pattern': 1.0,
                'streak': 1.0,
                'bias': 1.0,
                'alternating': 1.0,
                'banker_bias': 0.5  # Start lower as this is a fallback
            },
            'action_results': {
                'pattern': [],
                'streak': [],
                'bias': [],
                'alternating': [],
                'banker_bias': []
            },
            'pattern_stats': {},
            'streak_stats': {},
            'last_prediction': None,
            'last_action': None,
            'exploration_decay': 0.99,
            'current_exploration': self.exploration_rate
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with:"
                   f"\n\tLearning rate: {self.learning_rate}"
                   f"\n\tExploration rate: {self.exploration_rate}"
                   f"\n\tPattern length: {self.pattern_length}"
                   f"\n\tMin threshold: {self.min_threshold}"
                   f"\n\tAction history size: {self.action_history_size}")

    def get_bet(self, outcomes):
        """
        Determine next bet using dynamic adaptive learning
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
            action = state['last_action']
            
            # Record result for this action type
            if action:
                state['action_results'][action].append(1 if is_win else 0)
                if len(state['action_results'][action]) > self.action_history_size:
                    state['action_results'][action].pop(0)
                
                # Update weights based on action performance
                if len(state['action_results'][action]) >= 5:  # Need minimum samples
                    recent_performance = np.mean(state['action_results'][action][-5:])
                    
                    # Adjust weight using performance feedback
                    if is_win:
                        # Reinforce successful actions
                        state['action_weights'][action] *= (1 + self.learning_rate)
                    else:
                        # Reduce weight for unsuccessful actions
                        state['action_weights'][action] *= (1 - self.learning_rate)
                    
                    # Ensure weights stay in reasonable range
                    state['action_weights'][action] = max(0.1, min(5.0, state['action_weights'][action]))
            
            # Update pattern stats
            if len(historical_outcomes) >= self.pattern_length:
                pattern = ''.join(historical_outcomes[-(self.pattern_length+1):-1])
                if pattern not in state['pattern_stats']:
                    state['pattern_stats'][pattern] = {'P': 0, 'B': 0, 'total': 0}
                state['pattern_stats'][pattern][actual] += 1
                state['pattern_stats'][pattern]['total'] += 1
                
            # Update streak stats
            prev_streak_val = historical_outcomes[-2]
            streak_len = 1
            for i in range(len(historical_outcomes)-3, -1, -1):
                if historical_outcomes[i] == prev_streak_val:
                    streak_len += 1
                else:
                    break
            
            streak_key = f"{prev_streak_val}_{min(streak_len, 5)}"
            if streak_key not in state['streak_stats']:
                state['streak_stats'][streak_key] = {'P': 0, 'B': 0, 'total': 0}
            state['streak_stats'][streak_key][actual] += 1
            state['streak_stats'][streak_key]['total'] += 1
            
            # Decay exploration rate over time to exploit more
            state['current_exploration'] *= state['exploration_decay']
            state['current_exploration'] = max(0.05, state['current_exploration'])
        
        # Generate predictions from different approaches
        predictions = {}
        confidences = {}
        
        # 1. Pattern matching approach
        pattern = ''.join(historical_outcomes[-self.pattern_length:])
        pattern_pred, pattern_conf = self._get_pattern_prediction(pattern, historical_outcomes)
        predictions['pattern'] = pattern_pred
        confidences['pattern'] = pattern_conf * state['action_weights']['pattern']
        
        # 2. Streak analysis approach
        streak_val = historical_outcomes[-1]
        streak_len = 1
        for i in range(len(historical_outcomes)-2, -1, -1):
            if historical_outcomes[i] == streak_val:
                streak_len += 1
            else:
                break
                
        streak_pred, streak_conf = self._get_streak_prediction(streak_val, streak_len)
        predictions['streak'] = streak_pred
        confidences['streak'] = streak_conf * state['action_weights']['streak']
        
        # 3. Bias detection approach
        bias_pred, bias_conf = self._get_bias_prediction(historical_outcomes)
        predictions['bias'] = bias_pred
        confidences['bias'] = bias_conf * state['action_weights']['bias']
        
        # 4. Alternating pattern detection
        alt_pred, alt_conf = self._get_alternating_prediction(historical_outcomes)
        predictions['alternating'] = alt_pred
        confidences['alternating'] = alt_conf * state['action_weights']['alternating']
        
        # 5. Banker bias (fallback)
        predictions['banker_bias'] = 'B'
        confidences['banker_bias'] = 0.51 * state['action_weights']['banker_bias']  # Slight edge due to lower commission
        
        # Make decision - either explore or exploit
        if np.random.random() < state['current_exploration']:
            # Exploration - choose random action but weighted by performance
            weights = list(state['action_weights'].values())
            actions = list(state['action_weights'].keys())
            total_weight = sum(weights)
            
            if total_weight > 0:
                norm_weights = [w/total_weight for w in weights]
                chosen_action = np.random.choice(actions, p=norm_weights)
                chosen_prediction = predictions[chosen_action]
                state['last_action'] = chosen_action
            else:
                chosen_prediction = 'B'  # Default
                state['last_action'] = 'banker_bias'
                
            logger.debug(f"DYNAMIC_ADAPTIVE: Exploring with action {state['last_action']}")
        else:
            # Exploitation - choose action with highest weighted confidence
            best_confidence = -1
            best_action = None
            best_prediction = None
            
            for action, conf in confidences.items():
                if conf > best_confidence:
                    best_confidence = conf
                    best_action = action
                    best_prediction = predictions[action]
            
            # Only use prediction if confidence exceeds threshold
            if best_confidence >= self.min_threshold:
                chosen_prediction = best_prediction
                state['last_action'] = best_action
                logger.debug(f"DYNAMIC_ADAPTIVE: Exploiting action {best_action} with confidence {best_confidence:.2f}")
            else:
                # Default to banker if no confident prediction
                chosen_prediction = 'B'
                state['last_action'] = 'banker_bias'
                logger.debug(f"DYNAMIC_ADAPTIVE: No confident prediction, using banker bias")
        
        # Store prediction for next round
        state['last_prediction'] = chosen_prediction
        return chosen_prediction
    
    def _get_pattern_prediction(self, pattern, historical_outcomes):
        """Get prediction based on pattern matching"""
        state = self._state
        
        # Check if we have stats for this pattern
        if pattern in state['pattern_stats'] and state['pattern_stats'][pattern]['total'] >= 3:
            stats = state['pattern_stats'][pattern]
            p_prob = stats['P'] / stats['total']
            
            if p_prob > 0.6:
                return 'P', 0.5 + (p_prob - 0.5)
            elif p_prob < 0.4:
                return 'B', 0.5 + (0.5 - p_prob)
        
        # Search for pattern in history
        matches = []
        for i in range(len(historical_outcomes) - self.pattern_length * 2):
            past = historical_outcomes[i:i+self.pattern_length]
            current = historical_outcomes[-self.pattern_length:]
            if past == current and i + self.pattern_length < len(historical_outcomes):
                matches.append(historical_outcomes[i + self.pattern_length])
        
        if len(matches) >= 3:
            p_count = matches.count('P')
            p_prob = p_count / len(matches)
            confidence = 0.5 + min(0.3, 0.05 * len(matches))
            
            if p_prob > 0.6:
                return 'P', confidence
            elif p_prob < 0.4:
                return 'B', confidence
                
        return 'B', 0.5  # Default with neutral confidence
    
    def _get_streak_prediction(self, streak_val, streak_len):
        """Get prediction based on streak analysis"""
        state = self._state
        
        # Cap streak length for lookup
        capped_len = min(streak_len, 5)
        streak_key = f"{streak_val}_{capped_len}"
        
        # Check if we have stats for this streak pattern
        if streak_key in state['streak_stats'] and state['streak_stats'][streak_key]['total'] >= 3:
            stats = state['streak_stats'][streak_key]
            p_prob = stats['P'] / stats['total']
            
            confidence = 0.5 + min(0.3, 0.05 * stats['total'])
            
            if p_prob > 0.6:
                return 'P', confidence
            elif p_prob < 0.4:
                return 'B', confidence
        
        # Default streak approach - bet against long streaks
        if streak_len >= 3:
            confidence = 0.5 + min(0.3, 0.05 * streak_len)
            return 'P' if streak_val == 'B' else 'B', confidence
                
        return 'B', 0.5  # Default with neutral confidence
    
    def _get_bias_prediction(self, historical_outcomes):
        """Get prediction based on bias detection"""
        if len(historical_outcomes) < 10:
            return 'B', 0.5
            
        # Check multiple window sizes for bias
        windows = [10, 20, 40] if len(historical_outcomes) >= 40 else [10, 20] if len(historical_outcomes) >= 20 else [10]
        
        best_confidence = 0.5
        best_prediction = 'B'
        
        for window in windows:
            recent = historical_outcomes[-window:]
            p_count = recent.count('P')
            b_count = recent.count('B')
            total = p_count + b_count
            
            if total > 0:
                p_ratio = p_count / total
                
                # Detect strong bias and bet against it (mean reversion)
                if p_ratio > 0.65:
                    conf = 0.5 + min(0.3, (p_ratio - 0.5) * 2)
                    if conf > best_confidence:
                        best_confidence = conf
                        best_prediction = 'B'
                elif p_ratio < 0.35:
                    conf = 0.5 + min(0.3, (0.5 - p_ratio) * 2)
                    if conf > best_confidence:
                        best_confidence = conf
                        best_prediction = 'P'
                        
        return best_prediction, best_confidence
    
    def _get_alternating_prediction(self, historical_outcomes):
        """Get prediction based on alternating patterns"""
        if len(historical_outcomes) < 4:
            return 'B', 0.5
            
        # Check for alternating pattern in last 4 outcomes
        alternating = True
        for i in range(1, min(4, len(historical_outcomes))):
            if i < len(historical_outcomes) and historical_outcomes[-i] == historical_outcomes[-(i+1)]:
                alternating = False
                break
                
        if alternating and len(historical_outcomes) >= 4:
            # If alternating, predict the next in sequence
            confidence = 0.7  # Strong confidence in alternating patterns
            prediction = 'P' if historical_outcomes[-1] == 'B' else 'B'
            return prediction, confidence
            
        return 'B', 0.5  # Default with neutral confidence