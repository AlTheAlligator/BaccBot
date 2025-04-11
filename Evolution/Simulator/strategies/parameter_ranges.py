"""
Parameter ranges for strategy optimization.

This file defines the parameter ranges for each strategy that can be used
for parameter optimization. Each strategy has its own set of parameters
with defined ranges for testing.

Format for parameter ranges:
{
    'param_name': {
        'min': minimum_value,
        'max': maximum_value,
        'steps': number_of_steps_to_test
    },
    'boolean_param': {
        'values': [True, False]
    },
    'categorical_param': {
        'values': ['option1', 'option2', 'option3']
    }
}
"""

import logging

logger = logging.getLogger(__name__)

# Import the BettingStrategy enum
from .betting_strategy import BettingStrategy

# Dictionary mapping strategy enum to parameter ranges
STRATEGY_PARAMETER_RANGES = {
    # Original strategy
    BettingStrategy.ORIGINAL: {
        # Original strategy doesn't have parameters to optimize
    },

    # Follow Streak strategy
    BettingStrategy.FOLLOW_STREAK: {
        'streak_length': {'min': 2, 'max': 6, 'steps': 5}
    },

    # Counter Streak strategy
    BettingStrategy.COUNTER_STREAK: {
        'streak_length': {'min': 2, 'max': 6, 'steps': 5}
    },

    # Majority Last N strategy
    BettingStrategy.MAJORITY_LAST_N: {
        'n': {'min': 3, 'max': 30, 'steps': 28}
    },

    # Pattern Based strategy
    BettingStrategy.PATTERN_BASED: {
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5}
    },

    # Adaptive Bias strategy
    BettingStrategy.ADAPTIVE_BIAS: {
        'window_size': {'min': 5, 'max': 8, 'steps': 4},
        'weight_recent': {'min': 1, 'max': 10, 'steps': 50}
    },

    # Dynamic Adaptive strategy
    BettingStrategy.DYNAMIC_ADAPTIVE: {
        'pattern_length': {'min': 2, 'max': 15, 'steps': 14},
        'action_history_size': {'min': 5, 'max': 8, 'steps': 4},
        'min_threshold': {'min': 0.1, 'max': 0.7, 'steps': 20},
        'exploration_rate': {'min': 0.05, 'max': 0.7, 'steps': 50},
        'learning_rate': {'min': 0.05, 'max': 0.5, 'steps': 50}
    },

    # Hybrid Adaptive strategy
    BettingStrategy.HYBRID_ADAPTIVE: {
        'window_size': {'min': 5, 'max': 8, 'steps': 4},
        'weight_recent': {'min': 1, 'max': 10, 'steps': 50}
    },

    # Hybrid Pattern strategy
    BettingStrategy.HYBRID_PATTERN: {
        'pattern_length': {'min': 3, 'max': 6, 'steps': 5},
        'alternating_boost': {'min': 0.1, 'max': 10, 'steps': 50},
        'trend_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50}
    },

    # Hybrid Ensemble strategy
    BettingStrategy.HYBRID_ENSEMBLE: {
        'window_sizes': {'values': [[2, 3, 4], [2, 4, 6], [3, 4, 5], [3, 5, 6]]},
        'pattern_length': {'min': 2, 'max': 8, 'steps': 7},
        'streak_threshold': {'min': 2, 'max': 8, 'steps': 7},
        'pattern_weight': {'min': 0.1, 'max': 0.9, 'steps': 50},
        'confidence_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
    },

    # Hybrid ML strategy
    BettingStrategy.HYBRID_ML: {
        'window_size': {'min': 10, 'max': 50, 'steps': 5},
        'confidence_threshold': {'min': 0.5, 'max': 0.8, 'steps': 7},
        'learning_rate': {'min': 0.01, 'max': 0.2, 'steps': 5},
        'regularization': {'min': 0.001, 'max': 0.1, 'steps': 5},
        'use_feature_selection': {'values': [True, False]},
    },

    # Hybrid Majority strategy
    BettingStrategy.HYBRID_MAJORITY: {
        'feature_window': {'min': 2, 'max': 7, 'steps': 6},
        'learning_rate': {'min': 0.01, 'max': 0.6, 'steps': 50},
        'banker_bias': {'min': 0.25, 'max': 0.8, 'steps': 20},
        'min_confidence': {'min': 0.1, 'max': 0.9, 'steps': 50}
    },

    # Hybrid Simple Majority strategy
    BettingStrategy.HYBRID_SIMPLE_MAJORITY: {
        'window_size': {'min': 2, 'max': 7, 'steps': 6},
        'confidence_threshold': {'min': 0.2, 'max': 0.9, 'steps': 20}
    },

    # Enhanced Adaptive Bias strategy
    BettingStrategy.ENHANCED_ADAPTIVE_BIAS: {
        'window_sizes': {'values': [[2, 3, 4], [2, 4, 6], [3, 4, 5], [3, 5, 6]]},
        'base_weight_recent': {'min': 1, 'max': 5.0, 'steps': 50},
        'confidence_threshold': {'min': 0.2, 'max': 0.9, 'steps': 20},
        'adaptation_rate': {'min': 0.05, 'max': 0.5, 'steps': 20},
    },

    # Conservative Pattern strategy
    BettingStrategy.CONSERVATIVE_PATTERN: {
        'min_confidence': {'min': 0.3, 'max': 0.9, 'steps': 20},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'recovery_threshold': {'min': 1, 'max': 6, 'steps': 6},
        'skip_enabled': {'values': [False]},
    },

    # Loss Aversion strategy
    BettingStrategy.LOSS_AVERSION: {
        'base_confidence': {'min': 0.5, 'max': 0.7, 'steps': 5},
        'recovery_threshold': {'min': 1, 'max': 5, 'steps': 5},
        'window_size': {'min': 10, 'max': 30, 'steps': 5},
        'short_window': {'min': 3, 'max': 10, 'steps': 4},
        'banking_emphasis': {'min': 0.0, 'max': 0.2, 'steps': 5},
        'recovery_intensity': {'min': 0.5, 'max': 1.5, 'steps': 5},
        'max_skip_count': {'min': 0, 'max': 3, 'steps': 4},
        'pattern_boost': {'min': 0.0, 'max': 0.2, 'steps': 5},
        'adaptive_mode': {'values': [True, False]},
    },

    # Trend Confirmation strategy
    BettingStrategy.TREND_CONFIRMATION: {
        'window_sizes': {'values': [[2, 3, 4], [2, 4, 6], [3, 4, 5], [3, 5, 6]]},
        'min_threshold': {'min': 0.2, 'max': 0.9, 'steps': 50},
        'confirmation_threshold': {'min': 1, 'max': 5, 'steps': 5},
        'skip_enabled': {'values': [False]},
    },

    # Multi Condition strategy
    BettingStrategy.MULTI_CONDITION: {
        'window_size': {'min': 10, 'max': 30, 'steps': 5},
        'short_window': {'min': 3, 'max': 10, 'steps': 4},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'streak_threshold': {'min': 2, 'max': 5, 'steps': 4},
        'skip_enabled': {'values': [True, False]},
        'conditions_required': {'min': 1, 'max': 3, 'steps': 3},
    },

    # Dynamic Skip strategy
    BettingStrategy.DYNAMIC_SKIP: {
        'base_confidence': {'min': 0.5, 'max': 0.7, 'steps': 5},
        'risk_threshold': {'min': 0.3, 'max': 0.5, 'steps': 5},
        'window_size': {'min': 10, 'max': 30, 'steps': 5},
        'short_window': {'min': 3, 'max': 10, 'steps': 4},
        'recovery_factor': {'min': 0.01, 'max': 0.1, 'steps': 5},
    },

    # Selective Betting strategy
    BettingStrategy.SELECTIVE_BETTING: {
        'min_confidence': {'min': 0.6, 'max': 0.8, 'steps': 5},
        'ultra_confidence': {'min': 0.8, 'max': 0.95, 'steps': 4},
        'pattern_length': {'min': 3, 'max': 6, 'steps': 4},
        'window_sizes': {'values': [[5, 10, 20], [3, 10, 30], [5, 15, 30], [10, 20, 40]]},
        'losing_streak_threshold': {'min': 1, 'max': 4, 'steps': 4},
    },

    # Risk Parity strategy
    BettingStrategy.RISK_PARITY: {
        'lookback_window': {'min': 20, 'max': 50, 'steps': 4},
        'min_confidence': {'min': 0.55, 'max': 0.75, 'steps': 5},
        'max_volatility': {'min': 0.3, 'max': 0.7, 'steps': 5},
        'skip_ratio': {'min': 0.1, 'max': 0.5, 'steps': 5},
        'risk_limit': {'min': 1.0, 'max': 5.0, 'steps': 5},
    },

    # Streak Reversal Safe Exit strategy
    BettingStrategy.STREAK_REVERSAL_SAFE_EXIT: {
        'streak_threshold': {'min': 2, 'max': 5, 'steps': 4},
        'exit_loss_threshold': {'min': 1, 'max': 4, 'steps': 4},
        'recovery_period': {'min': 1, 'max': 5, 'steps': 5},
        'banker_bias': {'min': 0.5, 'max': 0.6, 'steps': 3},
        'window_size': {'min': 10, 'max': 30, 'steps': 5},
    },

    # Confidence Threshold Escalator strategy
    BettingStrategy.CONFIDENCE_THRESHOLD_ESCALATOR: {
        'base_threshold': {'min': 0.2, 'max': 0.9, 'steps': 50},
        'escalation_factor': {'min': 0.01, 'max': 0.2, 'steps': 50},
        'max_threshold': {'min': 0.4, 'max': 0.9, 'steps': 50},
        'de_escalation_factor': {'min': 0.01, 'max': 0.2, 'steps': 50},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'window_sizes': {'values': [[2, 3, 4], [2, 4, 6], [3, 4, 5], [3, 5, 6]]}
    },

    # Reinforcement Learning strategy
    BettingStrategy.REINFORCEMENT_LEARNING: {
        'learning_rate': {'min': 0.01, 'max': 0.3, 'steps': 50},
        'discount_factor': {'min': 0.5, 'max': 0.99, 'steps': 50},
        'exploration_rate': {'min': 0.01, 'max': 0.4, 'steps': 50},
        'exploration_decay': {'min': 0.7, 'max': 0.999, 'steps': 50},
        'min_exploration_rate': {'min': 0.001, 'max': 0.1, 'steps': 50},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'feature_type': {'values': ['pattern', 'frequency', 'combined']},
        'use_adaptive_learning': {'values': [True, False]},
        'skip_confidence_threshold': {'min': 0.1, 'max': 0.5, 'steps': 5},
    },

    # Bayesian Inference strategy
    BettingStrategy.BAYESIAN_INFERENCE: {
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'confidence_threshold': {'min': 0.25, 'max': 0.9, 'steps': 20},
        'prior_strength': {'min': 1.0, 'max': 5.0, 'steps': 50},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'banker_edge_adjustment': {'min': 0.005, 'max': 0.02, 'steps': 50},
        'use_recency_weighting': {'values': [True, False]},
        'recency_factor': {'min': 0.6, 'max': 0.99, 'steps': 20},
        'use_context_features': {'values': [True, False]},
        'feature_window': {'min': 2, 'max': 6, 'steps': 5},
    },

    # Markov Chain strategy
    BettingStrategy.MARKOV_CHAIN: {
        'order': {'min': 1, 'max': 4, 'steps': 4},
        'min_samples': {'min': 4, 'max': 10, 'steps': 7},
        'min_sequence_observations': {'min': 2, 'max': 5, 'steps': 4},
        'confidence_threshold': {'min': 0.5, 'max': 0.8, 'steps': 7},
        'banker_bias': {'min': 0.001, 'max': 0.02, 'steps': 10},
        'use_higher_order_fallback': {'values': [True, False]},
        'use_smoothing': {'values': [True, False]},
        'smoothing_factor': {'min': 0.1, 'max': 0.5, 'steps': 5},
        'use_adaptive_order': {'values': [True, False]},
        'max_order': {'min': 2, 'max': 5, 'steps': 4},
    },

    # Volatility Adaptive strategy
    BettingStrategy.VOLATILITY_ADAPTIVE: {
        'short_window': {'min': 2, 'max': 6, 'steps': 5},
        'medium_window': {'min': 5, 'max': 12, 'steps': 8},
        'long_window': {'min': 10, 'max': 60, 'steps': 51},
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'high_volatility_threshold': {'min': 0.3, 'max': 0.9, 'steps': 20},
        'low_volatility_threshold': {'min': 0.1, 'max': 0.5, 'steps': 20},
        'confidence_threshold_base': {'min': 0.3, 'max': 0.8, 'steps': 20},
        'confidence_scaling': {'min': 0.01, 'max': 0.5, 'steps': 50},
        'banker_bias': {'min': 0.005, 'max': 0.02, 'steps': 20},
        'use_adaptive_window': {'values': [True, False]},
        'statistical_mode': {'values': ['frequency', 'pattern', 'streak', 'combined']},
        'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
        'min_pattern_occurrences': {'min': 2, 'max': 6, 'steps': 5},
    },

    # Pattern Interruption strategy
    BettingStrategy.PATTERN_INTERRUPTION: {
        'pattern_window': {'min': 5, 'max': 15, 'steps': 3},
        'min_samples': {'min': 10, 'max': 30, 'steps': 5},
        'repeat_threshold': {'min': 2, 'max': 5, 'steps': 4},
        'confidence_threshold': {'min': 0.5, 'max': 0.8, 'steps': 7},
        'post_interruption_followup': {'values': [True, False]},
        'max_followup_hands': {'min': 1, 'max': 5, 'steps': 5},
        'banker_bias': {'min': 0.005, 'max': 0.02, 'steps': 4},
    },

    # Frequency Analysis strategy
    BettingStrategy.FREQUENCY_ANALYSIS: {
        'short_window': {'min': 1, 'max': 12, 'steps': 12},
        'medium_window': {'min': 4, 'max': 20, 'steps': 17},
        'long_window': {'min': 8, 'max': 40, 'steps': 33},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'confidence_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
        'pattern_length': {'min': 1, 'max': 10, 'steps': 10},
        'banker_bias': {'min': 0.0, 'max': 0.05, 'steps': 20},
        'use_trend_adjustment': {'values': [True, False]},
        'trend_weight': {'min': 0.01, 'max': 0.9, 'steps': 50},
        'use_pattern_adjustment': {'values': [True, False]},
        'pattern_weight': {'min': 0.01, 'max': 0.9, 'steps': 50},
        'use_chi_square': {'values': [True, False]},
        'significance_level': {'min': 0.01, 'max': 0.9, 'steps': 50},
        'clustering_method': {'values': ['multi_window', 'clustering_method']},
    },

    # Deep Q Network strategy
    BettingStrategy.DEEP_Q_NETWORK: {
        'learning_rate': {'min': 0.001, 'max': 0.1, 'steps': 5},
        'discount_factor': {'min': 0.8, 'max': 0.99, 'steps': 5},
        'exploration_rate': {'min': 0.05, 'max': 0.3, 'steps': 6},
        'batch_size': {'min': 16, 'max': 128, 'steps': 4},
        'memory_size': {'min': 100, 'max': 1000, 'steps': 5},
    },

    # Genetic Algorithm strategy
    BettingStrategy.GENETIC_ALGORITHM: {
        'population_size': {'min': 10, 'max': 500, 'steps': 50},
        'mutation_rate': {'min': 0.01, 'max': 0.2, 'steps': 10},
        'crossover_rate': {'min': 0.3, 'max': 0.9, 'steps': 6},
        'generations': {'min': 5, 'max': 50, 'steps': 10},
        'selection_pressure': {'min': 1, 'max': 4.0, 'steps': 4},
    },

    # Thompson Sampling strategy
    BettingStrategy.THOMPSON_SAMPLING: {
        'prior_alpha': {'min': 0.1, 'max': 4.0, 'steps': 50},
        'prior_beta': {'min': 0.1, 'max': 4.0, 'steps': 50},
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'confidence_threshold': {'min': 0.2, 'max': 0.9, 'steps': 50},
        'use_context': {'values': [True, False]},
        'context_length': {'min': 1, 'max': 6, 'steps': 6},
        'banker_bias': {'min': 0.001, 'max': 0.05, 'steps': 50},
        'exploration_weight': {'min': 0.1, 'max': 3, 'steps': 50},
        'min_context_observations': {'min': 2, 'max': 6, 'steps': 5},
        'discount_factor': {'min': 0.8, 'max': 0.999, 'steps': 50},
        'use_recency_weighting': {'values': [True, False]},
    },

    # Sequential Pattern Mining strategy
    BettingStrategy.SEQUENTIAL_PATTERN_MINING: {
        'min_samples': {'min': 15, 'max': 30, 'steps': 4},
        'min_pattern_length': {'min': 2, 'max': 4, 'steps': 3},
        'max_pattern_length': {'min': 4, 'max': 8, 'steps': 5},
        'confidence_threshold': {'min': 0.6, 'max': 0.8, 'steps': 5},
        'min_support': {'min': 2, 'max': 5, 'steps': 4},
        'banker_bias': {'min': 0.005, 'max': 0.02, 'steps': 4},
        'use_weighted_patterns': {'values': [True, False]},
        'recency_factor': {'min': 0.1, 'max': 0.5, 'steps': 5},
        'pattern_timeout': {'min': 30, 'max': 100, 'steps': 4},
        'use_confidence_scaling': {'values': [True, False]},
    },

    # Momentum Oscillator strategy
    BettingStrategy.MOMENTUM_OSCILLATOR: {
        'short_window': {'min': 2, 'max': 10, 'steps': 9},
        'long_window': {'min': 8, 'max': 30, 'steps': 23},
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'overbought_threshold': {'min': 30, 'max': 90, 'steps': 50},
        'oversold_threshold': {'min': 10, 'max': 60, 'steps': 50},
        'signal_line_period': {'min': 2, 'max': 5, 'steps': 4},
        'reversal_weight': {'min': 0.1, 'max': 3, 'steps': 50},
        'trend_weight': {'min': 0.1, 'max': 3, 'steps': 50},
        'confidence_threshold': {'min': 0.2, 'max': 0.7, 'steps': 50},
        'banker_bias': {'min': 0.001, 'max': 0.05, 'steps': 50},
        'use_stochastic': {'values': [True, False]},
    },

    # Recurrent Neural Network strategy
    BettingStrategy.RECURRENT_NEURAL_NETWORK: {
        'hidden_size': {'min': 8, 'max': 64, 'steps': 4},
        'num_layers': {'min': 1, 'max': 3, 'steps': 3},
        'learning_rate': {'min': 0.001, 'max': 0.1, 'steps': 5},
        'sequence_length': {'min': 5, 'max': 20, 'steps': 4},
        'batch_size': {'min': 8, 'max': 64, 'steps': 4},
    },

    # Chaos Theory strategy
    BettingStrategy.CHAOS_THEORY: {
        'embedding_dimension': {'min': 2, 'max': 5, 'steps': 4},
        'time_delay': {'min': 1, 'max': 7, 'steps': 7},
        'prediction_horizon': {'min': 1, 'max': 7, 'steps': 7},
        'num_neighbors': {'min': 3, 'max': 100, 'steps': 97},
        'min_samples': {'min': 3, 'max': 6, 'steps': 4},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 10},
    },

    # Information Theory strategy
    BettingStrategy.INFORMATION_THEORY: {
        'entropy_window': {'min': 2, 'max': 30, 'steps': 29},
        'mutual_info_lag': {'min': 1, 'max': 5, 'steps': 5},
        'entropy_threshold': {'min': 0.5, 'max': 0.95, 'steps': 20},
        'mutual_info_threshold': {'min': 0.01, 'max': 0.5, 'steps': 20},
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 10},
    },

    # Quantum-Inspired strategy
    BettingStrategy.QUANTUM_INSPIRED: {
        'num_qubits': {'min': 2, 'max': 10, 'steps': 9},
        'amplitude_boost': {'min': 1.0, 'max': 2.0, 'steps': 5},
        'interference_factor': {'min': 0.1, 'max': 0.5, 'steps': 5},
        'grover_iterations': {'min': 1, 'max': 3, 'steps': 3},
        'min_samples': {'min': 2, 'max': 6, 'steps': 4},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 10},
    },

    # Fractal Analysis strategy
    BettingStrategy.FRACTAL_ANALYSIS: {
        'min_scale': {'min': 3, 'max': 10, 'steps': 8},
        'max_scale': {'min': 30, 'max': 100, 'steps': 8},
        'hurst_threshold': {'min': 0.4, 'max': 0.8, 'steps': 5},
        'min_samples': {'min': 2, 'max': 6, 'steps': 8},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 10},
    },

    # Evolutionary Game Theory strategy
    BettingStrategy.EVOLUTIONARY_GAME_THEORY: {
        'population_size': {'min': 5, 'max': 100, 'steps': 20},
        'mutation_rate': {'min': 0.05, 'max': 0.3, 'steps': 6},
        'selection_pressure': {'min': 1.0, 'max': 2.0, 'steps': 5},
        'min_samples': {'min': 2, 'max': 6, 'steps': 5},
        'evolution_interval': {'min': 2, 'max': 20, 'steps': 19},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
    },

    # Hybrid Frequency-Volatility strategy
    BettingStrategy.HYBRID_FREQUENCY_VOLATILITY: {
        # Main hybrid parameters
        'performance_window': {'min': 3, 'max': 100, 'steps': 98},
        'min_confidence_diff': {'min': 0.01, 'max': 0.2, 'steps': 20},
        'performance_weight': {'min': 0.01, 'max': 0.9, 'steps': 50},
        'confidence_weight': {'min': 0.01, 'max': 0.9, 'steps': 50},

        # Nested frequency parameters
        'frequency_params': {
            'short_window': {'min': 6, 'max': 8, 'steps': 3},
            'medium_window': {'min': 10, 'max': 12, 'steps': 3},
            'long_window': {'min': 18, 'max': 20, 'steps': 3},
            'min_samples': {'min': 2, 'max': 6, 'steps': 5},
            'confidence_threshold': {'min': 0.1, 'max': 0.9, 'steps': 20},
            'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
            'banker_bias': {'min': 0.001, 'max': 0.05, 'steps': 20},
            'use_trend_adjustment': {'values': [True, False]},
            'trend_weight': {'min': 0.01, 'max': 0.9, 'steps': 20},
            'use_pattern_adjustment': {'values': [True, False]},
            'pattern_weight': {'min': 0.01, 'max': 0.9, 'steps': 20},
            'use_chi_square': {'values': [True, False]},
            'significance_level': {'min': 0.01, 'max': 0.9, 'steps': 20},
            'clustering_method': {'values': ['multi_window', 'clustering_method']},
        },

        # Nested volatility parameters
        'volatility_params': {
            'short_window': {'min': 2, 'max': 6, 'steps': 5},
            'medium_window': {'min': 5, 'max': 12, 'steps': 8},
            'long_window': {'min': 10, 'max': 60, 'steps': 51},
            'min_samples': {'min': 2, 'max': 6, 'steps': 5},
            'high_volatility_threshold': {'min': 0.3, 'max': 0.9, 'steps': 20},
            'low_volatility_threshold': {'min': 0.1, 'max': 0.5, 'steps': 20},
            'confidence_threshold_base': {'min': 0.3, 'max': 0.8, 'steps': 20},
            'confidence_scaling': {'min': 0.01, 'max': 0.5, 'steps': 20},
            'banker_bias': {'min': 0.005, 'max': 0.02, 'steps': 20},
            'use_adaptive_window': {'values': [True, False]},
            'statistical_mode': {'values': ['frequency', 'pattern', 'streak', 'combined']},
            'pattern_length': {'min': 2, 'max': 6, 'steps': 5},
            'min_pattern_occurrences': {'min': 2, 'max': 6, 'steps': 5},
        }
    },

    # Neural Oscillator strategy
    BettingStrategy.NEURAL_OSCILLATOR: {
        'num_oscillators': {'min': 2, 'max': 12, 'steps': 11},
        'coupling_strength': {'min': 0.1, 'max': 0.5, 'steps': 10},
        'adaptation_rate': {'min': 0.01, 'max': 0.4, 'steps': 50},
        'resonance_threshold': {'min': 0.1, 'max': 0.9, 'steps': 9},
        'phase_sync_threshold': {'min': 0.1, 'max': 0.9, 'steps': 9},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
    },

    # Adaptive Momentum strategy
    BettingStrategy.ADAPTIVE_MOMENTUM: {
        'window_sizes': {'values': [[5, 10, 20, 30, 50], [3, 7, 15, 25, 40], [5, 15, 30, 60, 100]]},
        'weight_adaptation_rate': {'min': 0.01, 'max': 0.4, 'steps': 50},
        'threshold_adaptation_rate': {'min': 0.01, 'max': 0.4, 'steps': 50},
        'mean_reversion_threshold': {'min': 0.1, 'max': 0.8, 'steps': 20},
        'initial_threshold': {'min': 0.1, 'max': 0.9, 'steps': 20},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
    },

    # Symbolic Dynamics strategy
    BettingStrategy.SYMBOLIC_DYNAMICS: {
        'symbol_length': {'min': 2, 'max': 6, 'steps': 5},
        'recurrence_threshold': {'min': 0.1, 'max': 0.9, 'steps': 20},
        'entropy_threshold': {'min': 0.1, 'max': 0.9, 'steps': 20},
        'forbidden_pattern_weight': {'min': 0.1, 'max': 0.9, 'steps': 20},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
    },

    # Bayesian Network strategy
    BettingStrategy.BAYESIAN_NETWORK: {
        'max_parents': {'min': 1, 'max': 19, 'steps': 10},
        'prior_strength': {'min': 0.1, 'max': 5.0, 'steps': 50},
        'confidence_threshold': {'min': 0.1, 'max': 0.9, 'steps': 20},
        'learning_rate': {'min': 0.01, 'max': 0.3, 'steps': 50},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
    },

    # Reinforcement Meta-Learning strategy
    BettingStrategy.REINFORCEMENT_META_LEARNING: {
        'initial_learning_rate': {'min': 0.01, 'max': 0.4, 'steps': 50},
        'exploration_rate': {'min': 0.01, 'max': 0.3, 'steps': 50},
        'exploration_decay': {'min': 0.9, 'max': 0.999, 'steps': 20},
        'min_exploration': {'min': 0.01, 'max': 0.1, 'steps': 20},
        'reward_discount': {'min': 0.5, 'max': 0.99, 'steps': 20},
        'min_samples': {'min': 2, 'max': 10, 'steps': 9},
        'banker_bias': {'min': 0.0, 'max': 0.1, 'steps': 20},
        'context_length': {'min': 2, 'max': 10, 'steps': 9},
    }
}

def get_parameter_ranges(strategy):
    """
    Get parameter ranges for a specific strategy.

    Args:
        strategy: The BettingStrategy enum value

    Returns:
        dict: Parameter ranges for the strategy
    """
    # Convert enum to string value if it's an enum
    strategy_key = strategy.value if hasattr(strategy, 'value') else str(strategy)

    # Try to find the strategy in the parameter ranges dictionary
    for key, value in STRATEGY_PARAMETER_RANGES.items():
        # Compare string values to handle different enum instances
        if hasattr(key, 'value') and key.value == strategy_key:
            # Validate nested parameter ranges
            for param_name, param_range in value.items():
                if isinstance(param_range, dict) and not ('values' in param_range or ('min' in param_range and 'max' in param_range and 'steps' in param_range)):
                    # This is a nested parameter structure, validate each nested parameter
                    for nested_param, nested_range in param_range.items():
                        if not ('values' in nested_range or ('min' in nested_range and 'max' in nested_range and 'steps' in nested_range)):
                            logger.error(f"Invalid nested parameter range for {param_name}.{nested_param} in strategy {strategy_key}")
                            return {}
            return value

    logger.warning(f"No parameter ranges defined for strategy {strategy}. Using empty dict.")
    return {}

