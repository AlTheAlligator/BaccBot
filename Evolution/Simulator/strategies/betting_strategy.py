"""
Betting Strategy Enum

This module defines the BettingStrategy enum used throughout the simulator.
It's kept in a separate file to avoid circular imports.
"""

from enum import Enum

class BettingStrategy(Enum):
    ORIGINAL = "original"  # Original BBB/PPP strategy
    FOLLOW_STREAK = "follow_streak"  # Bet on the side that's on a winning streak
    COUNTER_STREAK = "counter_streak"  # Bet against a winning streak
    MAJORITY_LAST_N = "majority_last_n"  # Bet based on majority in last N outcomes
    PATTERN_BASED = "pattern_based"  # Look for specific patterns
    ADAPTIVE_BIAS = "adaptive_bias"  # Adaptively learn table bias
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"  # Adaptive bias with dynamic parameter adjustment
    HYBRID_ADAPTIVE = "hybrid_adaptive"  # Original strategy with adaptive mode selection
    HYBRID_PATTERN = "hybrid_pattern"  # Original strategy with pattern-based mode selection
    HYBRID_ENSEMBLE = "hybrid_ensemble"  # Original strategy with ensemble method selection
    HYBRID_ML = "hybrid_ml"  # Original strategy with machine learning for mode selection
    HYBRID_MAJORITY = "hybrid_majority"  # Original strategy with majority-based mode selection
    HYBRID_SIMPLE_MAJORITY = "hybrid_simple_majority"  # Original strategy with simple majority-based mode selection
    ENHANCED_ADAPTIVE_BIAS = "enhanced_adaptive_bias"  # Enhanced version of adaptive bias
    CONSERVATIVE_PATTERN = "conservative_pattern"  # Only bet when pattern confidence is very high
    LOSS_AVERSION = "loss_aversion"  # Strategy that prioritizes avoiding consecutive losses
    TREND_CONFIRMATION = "trend_confirmation"  # Wait for multiple confirmations before betting
    MULTI_CONDITION = "multi_condition"  # Only bet when multiple conditions align
    DYNAMIC_SKIP = "dynamic_skip"  # Dynamically skip betting in uncertain situations
    SELECTIVE_BETTING = "selective_betting"  # Only bet on highest confidence opportunities
    RISK_PARITY = "risk_parity"  # Balance risk across different betting patterns
    STREAK_REVERSAL_SAFE_EXIT = "streak_reversal_safe_exit"  # Bet on every hand but exit safely after losses
    CONFIDENCE_THRESHOLD_ESCALATOR = "confidence_threshold_escalator"  # Dynamic confidence thresholds
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # Strategy that learns optimal betting through RL
    MARKOV_CHAIN = "markov_chain"  # Use Markov chain to predict next outcome based on transition probabilities
    BAYESIAN_INFERENCE = "bayesian_inference"  # Update outcome probabilities using Bayesian statistics
    MOMENTUM_OSCILLATOR = "momentum_oscillator"  # Technical analysis inspired oscillator to detect momentum shifts
    TIME_SERIES_FORECASTING = "time_series_forecasting"  # ARIMA or other time series forecasting methods
    ENSEMBLE_VOTING = "ensemble_voting"  # Combining multiple strategies with weighted voting
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"  # Run simulations to find optimal betting decisions
    META_STRATEGY = "meta_strategy"  # Strategy that selects among other strategies dynamically
    TRANSFER_LEARNING = "transfer_learning"  # Apply knowledge from previous shoes to current shoe
    VOLATILITY_ADAPTIVE = "volatility_adaptive"  # Adapt strategy based on detected table volatility
    PATTERN_INTERRUPTION = "pattern_interruption"  # Detect and bet on pattern interruptions
    FREQUENCY_ANALYSIS = "frequency_analysis"  # Analyze frequency distributions across time windows
    DEEP_Q_NETWORK = "deep_q_network"  # Deep Q-learning for optimal betting
    GENETIC_ALGORITHM = "genetic_algorithm"  # Evolve optimal betting strategies
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian exploration/exploitation trade-off
    SEQUENTIAL_PATTERN_MINING = "sequential_pattern_mining"  # Mine sequential patterns in outcomes
    RECURRENT_NEURAL_NETWORK = "recurrent_neural_network"  # RNN for sequence prediction
    CHAOS_THEORY = "chaos_theory"  # Look for deterministic patterns in seemingly random sequences
    INFORMATION_THEORY = "information_theory"  # Analyze entropy and mutual information in outcomes
    QUANTUM_INSPIRED = "quantum_inspired"  # Apply quantum computing concepts to decision making
    FRACTAL_ANALYSIS = "fractal_analysis"  # Look for self-similar patterns at different scales
    EVOLUTIONARY_GAME_THEORY = "evolutionary_game_theory"  # Model the game as a population of strategies
