"""
Baccarat Betting Strategies Package

This package contains modular implementations of various betting strategies for baccarat.
Each strategy inherits from the BaseStrategy class and implements specific betting logic.
"""

# Import all strategy classes for easier access
from .base_strategy import BaseStrategy
from .original import OriginalStrategy
from .follow_streak import FollowStreakStrategy
from .counter_streak import CounterStreakStrategy
from .majority_last_n import MajorityLastNStrategy
from .pattern_based import PatternBasedStrategy
from .adaptive_bias import AdaptiveBiasStrategy
from .dynamic_adaptive import DynamicAdaptiveStrategy
from .hybrid_adaptive import HybridAdaptiveStrategy
from .hybrid_pattern import HybridPatternStrategy
from .hybrid_ensemble import HybridEnsembleStrategy
from .hybrid_ml import HybridMLStrategy
from .hybrid_majority import HybridMajorityStrategy
from .hybrid_simple_majority import HybridSimpleMajorityStrategy
from .enhanced_adaptive_bias import EnhancedAdaptiveBiasStrategy
from .conservative_pattern import ConservativePatternStrategy
from .loss_aversion import LossAversionStrategy
from .trend_confirmation import TrendConfirmationStrategy
from .multi_condition import MultiConditionStrategy
from .dynamic_skip import DynamicSkipStrategy
from .selective_betting import SelectiveBettingStrategy
from .risk_parity import RiskParityStrategy
from .streak_reversal_safe_exit import StreakReversalSafeExitStrategy
from .confidence_threshold_escalator import ConfidenceThresholdEscalatorStrategy
from .deep_q_network import DeepQNetworkStrategy
from .genetic_algorithm import GeneticAlgorithmStrategy
from .time_series_forecasting import TimeSeriesForecastingStrategy
from .monte_carlo_simulation import MonteCarloSimulationStrategy
from .transfer_learning import TransferLearningStrategy
from .chaos_theory import ChaosTheoryStrategy
from .information_theory import InformationTheoryStrategy
from .quantum_inspired import QuantumInspiredStrategy
from .fractal_analysis import FractalAnalysisStrategy
from .evolutionary_game_theory import EvolutionaryGameTheoryStrategy
from .neural_oscillator import NeuralOscillatorStrategy
from .adaptive_momentum import AdaptiveMomentumStrategy
from .symbolic_dynamics import SymbolicDynamicsStrategy
from .bayesian_network import BayesianNetworkStrategy
from .reinforcement_meta_learning import ReinforcementMetaLearningStrategy

# Import the BettingStrategy enum
from .betting_strategy import BettingStrategy

# Import parameter ranges for strategy optimization
from .parameter_ranges import get_parameter_ranges