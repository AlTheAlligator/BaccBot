"""
Optimized Parameters for Baccarat Strategies.

This module contains pre-optimized parameter sets for different baccarat betting strategies.
These parameters have been fine-tuned through extensive simulation and optimization to achieve
the best performance in terms of win rate, profit, and risk management.
"""

import logging

logger = logging.getLogger(__name__)

# Dictionary of optimized parameters for each strategy
OPTIMIZED_PARAMETERS = {
    # Meta Strategy - dynamically selects among other strategies
    "meta_strategy": {
        "min_samples": 18,
        "confidence_threshold": 0.64,
        "banker_bias": 0.012,
        "use_weighted_voting": True,
        "adaptation_rate": 0.12,
        "exploration_factor": 0.03,
        "strategies": [
            "frequency_analysis",
            "sequential_pattern_mining", 
            "volatility_adaptive",
            "thompson_sampling",
            "pattern_interruption"
        ],
        # Parameters for sub-strategies
        "frequency_analysis": {
            "confidence_threshold": 0.59
        },
        "sequential_pattern_mining": {
            "confidence_threshold": 0.66
        },
        "volatility_adaptive": {
            "confidence_threshold_base": 0.56
        },
        "thompson_sampling": {
            "confidence_threshold": 0.56
        },
        "pattern_interruption": {
            "confidence_threshold": 0.61
        }
    },
    
    # Markov Chain Strategy - transition probability matrix based prediction
    "markov_chain": {
        "order": 3,
        "min_samples": 6,
        "min_sequence_observations": 4,
        "confidence_threshold": 0.62,
        "banker_bias": 0.011,
        "use_higher_order_fallback": True,
        "use_smoothing": True,
        "smoothing_factor": 0.4,
        "use_adaptive_order": True,
        "max_order": 4
    },
    
    # Thompson Sampling Strategy - Bayesian multi-armed bandit approach
    "thompson_sampling": {
        "prior_alpha": 1.2,
        "prior_beta": 1.2,
        "min_samples": 6,
        "confidence_threshold": 0.56,
        "use_context": True,
        "context_length": 3,
        "banker_bias": 0.012,
        "exploration_weight": 0.95,
        "min_context_observations": 2,
        "discount_factor": 0.98,
        "use_recency_weighting": True
    },
    
    # Pattern Interruption Strategy - detects when patterns break
    "pattern_interruption": {
        "pattern_window": 6,
        "min_samples": 8,
        "repeat_threshold": 2,
        "confidence_threshold": 0.61,
        "post_interruption_followup": True,
        "max_followup_hands": 2,
        "banker_bias": 0.011
    },
    
    # Frequency Analysis Strategy - analyzes outcome distributions
    "frequency_analysis": {
        "short_window": 7,
        "medium_window": 11,
        "long_window": 19,
        "min_samples": 6,
        "confidence_threshold": 0.2285,
        "pattern_length": 4,
        "banker_bias": 0.0164,
        "use_trend_adjustment": False,
        "trend_weight": 0.4095,
        "use_pattern_adjustment": True,
        "pattern_weight": 0.2279,
        "use_chi_square": False,
        "significance_level": 0.8455,
        "clustering_method": "multi_window"
    },
    
    # Sequential Pattern Mining Strategy - discovers frequent sequential patterns
    "sequential_pattern_mining": {
        "min_samples": 6,
        "min_pattern_length": 2,
        "max_pattern_length": 5,
        "confidence_threshold": 0.66,
        "min_support": 3,
        "banker_bias": 0.011,
        "use_weighted_patterns": True,
        "recency_factor": 0.15,
        "pattern_timeout": 60,
        "use_confidence_scaling": True
    },
    
    # Volatility Adaptive Strategy - adjusts approach based on volatility
    "volatility_adaptive": {
        'short_window': 5, 'medium_window': 10, 'long_window': 56, 'min_samples': 6, 'high_volatility_threshold': 0.5842105263157895, 'low_volatility_threshold': 0.22631578947368422, 'confidence_threshold_base': 0.6105263157894738, 'confidence_scaling': 0.5, 'banker_bias': 0.016938775510204084, 'use_adaptive_window': False, 'statistical_mode': 'combined', 'pattern_length': 5, 'min_pattern_occurrences': 3}
    ,
    
    # Bayesian Inference Strategy - updates probabilities with Bayes' rule
    "bayesian_inference": {
        "min_samples": 6,
        "confidence_threshold": 0.55,
        "prior_strength": 3.0,
        "pattern_length": 3,
        "banker_edge_adjustment": 0.011,
        "use_recency_weighting": True,
        "recency_factor": 0.95,
        "use_context_features": True,
        "feature_window": 4
    },
    
    # Momentum Oscillator Strategy - technical analysis inspired
    "momentum_oscillator": {
        "short_window": 3,
        "long_window": 6,
        "min_samples": 6,
        "overbought_threshold": 68,
        "oversold_threshold": 32,
        "signal_line_period": 3,
        "reversal_weight": 1.1,
        "trend_weight": 0.9,
        "confidence_threshold": 0.61,
        "banker_bias": 0.011,
        "use_stochastic": True
    },

    # Adaptive Bias Strategy
    "adaptive_bias": {
        "window_size": 8,
        "weight_recent": 2.0,
        "min_samples": 6,
        "confidence_threshold": 0.58,
        "banker_bias": 0.011
    },
    
    # Enhanced Adaptive Bias Strategy
    "enhanced_adaptive_bias": {
        "window_sizes": [4, 5, 6],
        "base_weight_recent": 2.5,
        "min_samples": 6,
        "confidence_threshold": 0.60,
        "pattern_length": 4,
        "use_pattern_recognition": True,
        "use_trend_analysis": True,
        "banker_bias": 0.011
    },
    
    # Selective Betting Strategy
    "selective_betting": {
        "min_confidence": 0.70,
        "ultra_confidence": 0.85,
        "pattern_length": 4,
        "window_sizes": [4, 5, 6],
        "losing_streak_threshold": 2,
        "banker_bias": 0.011
    },
    
    # Multi Condition Strategy
    "multi_condition": {
        "window_size": 6,
        "short_window": 5,
        "pattern_length": 4,
        "streak_threshold": 3,
        "skip_enabled": False,
        "conditions_required": 2,
        "confidence_threshold": 0.62,
        "banker_bias": 0.011
    },
    
    # Dynamic Skip Strategy
    "dynamic_skip": {
        "window_size": 6,
        "short_window": 5,
        "pattern_length": 4,
        "min_confidence": 0.65,
        "skip_threshold": 0.55,
        "recovery_threshold": 2,
        "banker_bias": 0.011
    },
    
    # Conservative Pattern Strategy
    "conservative_pattern": {
        "min_confidence": 0.70,
        "pattern_length": 4,
        "recovery_threshold": 2,
        "skip_enabled": False,
        "banker_bias": 0.011
    },
    
    # Trend Confirmation Strategy
    "trend_confirmation": {
        "window_sizes": [4, 5, 6],
        "min_threshold": 0.60,
        "confirmation_threshold": 2,
        "skip_enabled": False,
        "banker_bias": 0.011
    },
    
    # Confidence Threshold Escalator Strategy
    "confidence_threshold_escalator": {
        "base_threshold": 0.55,
        "escalation_factor": 0.05,
        "max_threshold": 0.85,
        "de_escalation_factor": 0.02,
        "pattern_length": 4,
        "window_sizes": [4, 5, 6],
        "banker_bias": 0.011
    },
    # Add to OPTIMIZED_PARAMETERS dictionary
    "hybrid_frequency_volatility": {
        'performance_window': 17, 
        'min_confidence_diff': 0.14, 
        'performance_weight': 0.28244897959183674, 
        'confidence_weight': 0.3914285714285714, 
        'frequency_params': 
        {
            'short_window': 7, 
            'medium_window': 11, 
            'long_window': 19, 
            'min_samples': 6, 
            'confidence_threshold': 0.31052631578947365, 
            'pattern_length': 2, 
            'banker_bias': 0.001, 
            'use_trend_adjustment': True, 
            'trend_weight': 0.9, 
            'use_pattern_adjustment': False, 
            'pattern_weight': 0.8531578947368421, 
            'use_chi_square': True, 
            'significance_level': 0.10368421052631578, 
            'clustering_method': 'clustering_method'
        }, 
        'volatility_params': 
        {
            'short_window': 3, 
            'medium_window': 8, 
            'long_window': 18, 
            'min_samples': 5, 
            'high_volatility_threshold': 0.5210526315789474, 
            'low_volatility_threshold': 0.4789473684210527, 
            'confidence_threshold_base': 0.3263157894736842, 
            'confidence_scaling': 0.5, 
            'banker_bias': 0.02, 
            'use_adaptive_window': True, 
            'statistical_mode': 'frequency', 
            'pattern_length': 6, 
            'min_pattern_occurrences': 5
        }
    },
    "neural_oscillator": 
    {
        'num_oscillators': 10, 
        'coupling_strength': 0.17346938775510204, 
        'adaptation_rate': 0.5638775510204082, 
        'resonance_threshold': 0.37755102040816324, 
        'phase_sync_threshold': 0.3122448979591837, 
        'min_samples': 4, 
        'banker_bias': 0.10526315789473684, 
        'phases': [3.497356267915043, 2.079982724598674, 1.5025799854356623, 3.0395457838984394, 4.274668208450234, 1.0993834171549892, 1.9051589295570905, 4.637798869350713, 3.130793366247823, 2.255587705529264]
    }
,
}

# Ensemble strategy that combines multiple meta-strategies
ENSEMBLE_PARAMETERS = {
    "min_samples": 6,
    "confidence_threshold": 0.67,
    "banker_bias": 0.012,
    "use_weighted_voting": True,
    "meta_strategies": [
        # Each entry is a meta-strategy with its own sub-strategies
        {
            "name": "meta_frequency",
            "weight": 1.0,
            "strategies": [
                "frequency_analysis",
                "bayesian_inference",
                "thompson_sampling"
            ]
        },
        {
            "name": "meta_pattern",
            "weight": 1.0,
            "strategies": [
                "sequential_pattern_mining",
                "pattern_interruption",
                "markov_chain"
            ]
        },
        {
            "name": "meta_adaptive",
            "weight": 1.0,
            "strategies": [
                "volatility_adaptive",
                "momentum_oscillator",
                "thompson_sampling"
            ]
        }
    ]
}

# Conservative parameter sets with higher confidence thresholds
# These make fewer bets but with higher confidence
CONSERVATIVE_PARAMETERS = {
    "meta_strategy": {
        **OPTIMIZED_PARAMETERS["meta_strategy"],
        "confidence_threshold": 0.70,
        "min_samples": 6
    },
    "markov_chain": {
        **OPTIMIZED_PARAMETERS["markov_chain"],
        "confidence_threshold": 0.68,
        "min_sequence_observations": 5
    },
    "thompson_sampling": {
        **OPTIMIZED_PARAMETERS["thompson_sampling"],
        "confidence_threshold": 0.62
    },
    "pattern_interruption": {
        **OPTIMIZED_PARAMETERS["pattern_interruption"],
        "confidence_threshold": 0.67
    },
    "frequency_analysis": {
        **OPTIMIZED_PARAMETERS["frequency_analysis"],
        "confidence_threshold": 0.65
    },
    "sequential_pattern_mining": {
        **OPTIMIZED_PARAMETERS["sequential_pattern_mining"],
        "confidence_threshold": 0.72,
        "min_support": 4
    },
    "volatility_adaptive": {
        **OPTIMIZED_PARAMETERS["volatility_adaptive"],
        "confidence_threshold_base": 0.62
    },
    "bayesian_inference": {
        **OPTIMIZED_PARAMETERS["bayesian_inference"],
        "confidence_threshold": 0.61
    },
    "momentum_oscillator": {
        **OPTIMIZED_PARAMETERS["momentum_oscillator"],
        "short_window": 4,
        "long_window": 7,
        "confidence_threshold": 0.67
    },
    "adaptive_bias": {
        **OPTIMIZED_PARAMETERS["adaptive_bias"],
        "confidence_threshold": 0.65,
        "window_size": 6
    },
    "enhanced_adaptive_bias": {
        **OPTIMIZED_PARAMETERS["enhanced_adaptive_bias"],
        "confidence_threshold": 0.67,
        "window_sizes": [4, 5, 6]
    },
    "selective_betting": {
        **OPTIMIZED_PARAMETERS["selective_betting"],
        "min_confidence": 0.75,
        "ultra_confidence": 0.90
    },
    "multi_condition": {
        **OPTIMIZED_PARAMETERS["multi_condition"],
        "confidence_threshold": 0.68,
        "conditions_required": 3
    },
    "dynamic_skip": {
        **OPTIMIZED_PARAMETERS["dynamic_skip"],
        "min_confidence": 0.70,
        "skip_threshold": 0.60
    },
    "conservative_pattern": {
        **OPTIMIZED_PARAMETERS["conservative_pattern"],
        "min_confidence": 0.75
    },
    "trend_confirmation": {
        **OPTIMIZED_PARAMETERS["trend_confirmation"],
        "min_threshold": 0.65,
        "confirmation_threshold": 3
    },
    "confidence_threshold_escalator": {
        **OPTIMIZED_PARAMETERS["confidence_threshold_escalator"],
        "base_threshold": 0.60,
        "max_threshold": 0.90
    }
}

# Aggressive parameter sets with lower confidence thresholds
# These make more bets but with lower average confidence
AGGRESSIVE_PARAMETERS = {
    "meta_strategy": {
        **OPTIMIZED_PARAMETERS["meta_strategy"],
        "confidence_threshold": 0.58,
        "min_samples": 6
    },
    "markov_chain": {
        **OPTIMIZED_PARAMETERS["markov_chain"],
        "confidence_threshold": 0.56,
        "min_sequence_observations": 3
    },
    "thompson_sampling": {
        **OPTIMIZED_PARAMETERS["thompson_sampling"],
        "confidence_threshold": 0.51
    },
    "pattern_interruption": {
        **OPTIMIZED_PARAMETERS["pattern_interruption"],
        "confidence_threshold": 0.55
    },
    "frequency_analysis": {
        **OPTIMIZED_PARAMETERS["frequency_analysis"],
        "confidence_threshold": 0.53
    },
    "sequential_pattern_mining": {
        **OPTIMIZED_PARAMETERS["sequential_pattern_mining"],
        "confidence_threshold": 0.60,
        "min_support": 2
    },
    "volatility_adaptive": {
        **OPTIMIZED_PARAMETERS["volatility_adaptive"],
        "confidence_threshold_base": 0.51
    },
    "bayesian_inference": {
        **OPTIMIZED_PARAMETERS["bayesian_inference"],
        "confidence_threshold": 0.50
    },
    "momentum_oscillator": {
        **OPTIMIZED_PARAMETERS["momentum_oscillator"],
        "confidence_threshold": 0.55
    },
    "adaptive_bias": {
        **OPTIMIZED_PARAMETERS["adaptive_bias"],
        "confidence_threshold": 0.52,
        "window_size": 6
    },
    "enhanced_adaptive_bias": {
        **OPTIMIZED_PARAMETERS["enhanced_adaptive_bias"],
        "confidence_threshold": 0.54,
        "window_sizes": [4, 5, 6]
    },
    "selective_betting": {
        **OPTIMIZED_PARAMETERS["selective_betting"],
        "min_confidence": 0.65,
        "ultra_confidence": 0.80
    },
    "multi_condition": {
        **OPTIMIZED_PARAMETERS["multi_condition"],
        "confidence_threshold": 0.56,
        "conditions_required": 1
    },
    "dynamic_skip": {
        **OPTIMIZED_PARAMETERS["dynamic_skip"],
        "min_confidence": 0.60,
        "skip_threshold": 0.50
    },
    "conservative_pattern": {
        **OPTIMIZED_PARAMETERS["conservative_pattern"],
        "min_confidence": 0.65,
        "skip_enabled": False
    },
    "trend_confirmation": {
        **OPTIMIZED_PARAMETERS["trend_confirmation"],
        "min_threshold": 0.55,
        "confirmation_threshold": 1
    },
    "confidence_threshold_escalator": {
        **OPTIMIZED_PARAMETERS["confidence_threshold_escalator"],
        "base_threshold": 0.50,
        "max_threshold": 0.80
    }
}

def get_parameters(strategy_name, parameter_set="optimized"):
    """
    Get the parameter set for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        parameter_set: Which set to use ("optimized", "conservative", or "aggressive")
        
    Returns:
        dict: Parameters for the strategy
    """
    if parameter_set == "conservative":
        params = CONSERVATIVE_PARAMETERS.get(strategy_name)
        if params:
            return params
        # Fall back to optimized if no conservative params exist
        logger.warning(f"No conservative parameters found for {strategy_name}, using optimized")
        
    elif parameter_set == "aggressive":
        params = AGGRESSIVE_PARAMETERS.get(strategy_name)
        if params:
            return params
        # Fall back to optimized if no aggressive params exist
        logger.warning(f"No aggressive parameters found for {strategy_name}, using optimized")
    
    # Default to optimized parameters
    params = OPTIMIZED_PARAMETERS.get(strategy_name)
    
    if not params:
        logger.warning(f"No parameters found for {strategy_name}, using empty dict")
        return {}
        
    return params

def get_ensemble_parameters():
    """Get the ensemble strategy parameters."""
    return ENSEMBLE_PARAMETERS
