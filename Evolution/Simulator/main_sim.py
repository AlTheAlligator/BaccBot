from enum import Enum
import random
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Seed for reproducibility
random.seed(42)

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
    PATTERN_INTERRUPTION = "pattern_interruption"  # Bet when established patterns are interrupted
    FREQUENCY_ANALYSIS = "frequency_analysis"  # Analyze frequency distributions of outcomes
    DEEP_Q_NETWORK = "deep_q_network"  # Deep reinforcement learning with neural networks
    GENETIC_ALGORITHM = "genetic_algorithm"  # Evolve optimal betting strategies
    THOMPSON_SAMPLING = "thompson_sampling"  # Multi-armed bandit approach with Bayesian updates
    SEQUENTIAL_PATTERN_MINING = "sequential_pattern_mining"  # Discover frequent sequential patterns
    RECURRENT_NEURAL_NETWORK = "recurrent_neural_network"  # RNN-based model to predict outcomes

class Bet:
    """A class representing a single bet in baccarat."""
    
    VALID_SIDES = {'P', 'B', 'SKIP'}
    VALID_RESULTS = {'W', 'L', 'T'}

    def __init__(self, side, size=1):
        """Initialize a bet.
        
        Args:
            side: 'P' for Player, 'B' for Banker, 'SKIP' to skip betting
            size: Bet size (default 1 unit)
            
        Raises:
            ValueError: If side is not valid
        """
        if side not in self.VALID_SIDES:
            raise ValueError(f"Invalid bet side: {side}. Must be one of: {self.VALID_SIDES}")
            
        self.side = side
        self.size = size
        self.result = None
        self.profit = 0
        
    def set_result(self, result):
        """Set the result of the bet and calculate profit/loss.
        
        Args:
            result: 'W' for win, 'L' for loss, 'T' for tie
            
        Returns:
            float: The profit/loss from this bet
            
        Raises:
            ValueError: If result is not valid
            RuntimeError: If result is set multiple times
        """
        if result not in self.VALID_RESULTS:
            raise ValueError(f"Invalid result: {result}. Must be one of: {self.VALID_RESULTS}")
            
        if self.result is not None:
            raise RuntimeError("Cannot set result multiple times")
            
        self.result = result
        
        if result == 'W':
            # Account for commission on banker bets
            self.profit = self.size * (0.95 if self.side == 'B' else 1)
        elif result == 'L':
            self.profit = -self.size
        else:  # Tie
            self.profit = 0
            
        return self.profit
        
    def __str__(self):
        """String representation of the bet."""
        result_str = self.result if self.result else 'Pending'
        return f"Bet({self.side}, size={self.size}, result={result_str}, profit={self.profit})"

class GameSimulator:
    def __init__(self, outcomes, initial_mode, strategy=BettingStrategy.ORIGINAL, strategy_params=None):
        """
        Initialize the simulator with the given parameters.
        
        Args:
            outcomes: List of initial outcomes
            initial_mode: Starting mode ('PPP' or 'BBB')
            strategy: Strategy enum value to use
            strategy_params: Optional parameters for the strategy
            
        Raises:
            ValueError: If parameters are invalid
        """
        if initial_mode not in ["PPP", "BBB"]:
            raise ValueError("initial_mode must be 'PPP' or 'BBB'")
            
        if not isinstance(strategy, BettingStrategy):
            raise ValueError("strategy must be a BettingStrategy enum value")
            
        if strategy_params is not None and not isinstance(strategy_params, dict):
            raise ValueError("strategy_params must be a dictionary or None")
            
        self.outcomes = []
        self.processed_outcomes = []
        self.initial_mode = initial_mode
        self.current_mode = initial_mode
        self.last_bet = None
        self.strategy_type = strategy
        self.strategy_params = strategy_params if strategy_params else {}
        # Use a constant bet size for flat betting
        self.bet_size = 10  # Default flat bet size
        
        # Override bet size if provided in strategy_params
        if strategy_params and 'bet_size' in strategy_params:
            self.bet_size = strategy_params['bet_size']
        
        try:
            self.strategy = self._create_strategy_instance()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to create strategy instance: {e}")
        
    def _create_strategy_instance(self):
        """
        Create an instance of the selected strategy class.
        
        Returns:
            BaseStrategy: Instance of the selected strategy
            
        Raises:
            ImportError: If strategy module cannot be imported
            AttributeError: If strategy class is not found
        """
        strategy_module = __import__(f'strategies.{self.strategy_type.value}', fromlist=[''])
        class_name = ''.join(word.capitalize() for word in self.strategy_type.value.split('_')) + 'Strategy'
        strategy_class = getattr(strategy_module, class_name)
        return strategy_class(self, self.strategy_params)

    def simulate_single(self, new_outcome=None):
        """
        Simulate a single round of betting.
        
        Args:
            new_outcome: Optional new outcome to add before getting next bet
            
        Returns:
            tuple: (next_bet, current_mode)
        """
        if new_outcome is not None:
            if new_outcome not in ['P', 'B', 'T']:
                raise ValueError("new_outcome must be 'P', 'B', or 'T'")
            self.outcomes.append(new_outcome)
            
        next_bet = self.get_strategy_bet(len(self.processed_outcomes))
        return next_bet, self.current_mode

    def get_strategy_bet(self, current_index):
        """
        Get the next bet from the current strategy.
        
        Args:
            current_index: Index in outcomes list to consider up to
            
        Returns:
            str: Next bet ('P', 'B', or 'SKIP')
            
        Raises:
            IndexError: If current_index is out of range
        """
        if current_index < 0 or current_index > len(self.outcomes):
            raise IndexError("current_index out of range")
            
        strategy_bet = self.strategy.get_bet(self.outcomes[:current_index])
        
        # Handle hybrid strategies that return None to indicate using original mode-based logic
        if strategy_bet is None:
            # Use original mode-based logic
            next_bet = "P" if self.current_mode == "BBB" else "B"
            return next_bet
            
        return strategy_bet
        
    def get_current_bet_size(self):
        """
        Get the bet size. With flat betting, this always returns the same amount.
        
        Returns:
            float: Current bet size
        """
        return self.bet_size

def get_optimized_strategy_parameters():
    """
    Get optimized parameter combinations for each strategy from optimized_parameters.py.
    This acts as a bridge to convert the optimized parameters to the format expected by the simulator.
    
    Returns:
        dict: Mapping of BettingStrategy enum values to lists of parameter sets
    """
    from strategies.optimized_parameters import OPTIMIZED_PARAMETERS, CONSERVATIVE_PARAMETERS, AGGRESSIVE_PARAMETERS
    
    # Convert strategy name strings to BettingStrategy enum values and format parameters
    strategy_params = {}
    
    for enum_val in BettingStrategy:
        # Convert enum value to parameter key (e.g., MOMENTUM_OSCILLATOR -> "momentum_oscillator")
        param_key = enum_val.value.lower()
        
        # Get parameters for this strategy from each set if they exist
        param_sets = []
        
        # Add optimized parameters if they exist
        if param_key in OPTIMIZED_PARAMETERS:
            params = OPTIMIZED_PARAMETERS[param_key].copy()
            params['description'] = 'Optimized parameters'
            param_sets.append(params)
            
        # Add conservative parameters if they exist
        if param_key in CONSERVATIVE_PARAMETERS:
            params = CONSERVATIVE_PARAMETERS[param_key].copy()
            params['description'] = 'Conservative parameters'
            param_sets.append(params)
            
        # Add aggressive parameters if they exist
        if param_key in AGGRESSIVE_PARAMETERS:
            params = AGGRESSIVE_PARAMETERS[param_key].copy()
            params['description'] = 'Aggressive parameters'
            param_sets.append(params)
            
        # Only add to strategy_params if we found parameters
        if param_sets:
            strategy_params[enum_val] = param_sets
            
    return strategy_params

def get_initial_outcomes_count(outcomes):
    """
    Determine how many initial outcomes to skip based on ties in first 6.
    
    Args:
        outcomes: List of outcomes ('P', 'B', 'T')
        
    Returns:
        int: Number of outcomes to skip (6 if no ties in first 6, 7 if there are ties)
    """
    if len(outcomes) < 6:
        return 6  # Default to 6 if we don't have enough outcomes
        
    # Count ties in first 6 outcomes
    ties_in_first_six = outcomes[:6].count('T')
    return 7 if ties_in_first_six > 0 else 6

def play_mode(current_mode, outcomes, initial_mode, last_bet, four_start):
    """
    Determine the next bet and handle switching between modes.
    """
    # Handle first bet or no last bet case
    if last_bet is None:
        return 'B' if current_mode == "PPP" else 'P', current_mode

    if current_mode == "Switch":
        # In switch mode, we alternate between P and B after a loss
        next_bet = 'B' if last_bet.side == 'P' else 'P'
        
        # Check if switching should stop (no more 3 consecutive)
        if not find_3_consecutive_losses(outcomes, initial_mode, four_start):
            updated_mode = initial_mode  # Return to original mode
            next_bet = 'B' if initial_mode == "PPP" else 'P'  # Bet opposite of mode
        else:
            updated_mode = "Switch"  # Stay in switch mode
    else:
        # In PPP or BBB mode, we bet opposite of the mode
        next_bet = 'B' if current_mode == "PPP" else 'P'
        
        # Check if we need to switch modes (found 3 consecutive)
        if find_3_consecutive_losses(outcomes, initial_mode, four_start):
            updated_mode = "Switch"  # Switch to alternating mode
            # Keep betting opposite of last result
            next_bet = 'B' if last_bet.side == 'P' else 'P'
        else:
            updated_mode = current_mode  # Stay in current mode

    return next_bet, updated_mode

def find_3_consecutive_losses(outcomes, bias, four_start):
    """
    Find 3 consecutive losses in the last 6 outcomes.
    Returns True if 3 consecutive of the SAME outcome as our bias are found.
    
    Args:
        outcomes: List of outcomes ('P', 'B', 'T')
        bias: String indicating the bias ('PPP' or 'BBB')
        four_start: Whether this started with 4 of the same outcome
        
    Returns:
        bool: True if 3 consecutive losses found
        
    Raises:
        ValueError: If bias is invalid or outcomes contains invalid values
    """
    if not outcomes:
        return False
        
    if bias not in ["PPP", "BBB"]:
        raise ValueError("bias must be 'PPP' or 'BBB'")
        
    if any(o not in ['P', 'B', 'T'] for o in outcomes):
        raise ValueError("outcomes must only contain 'P', 'B', or 'T'")
        
    if len(outcomes) < 6:
        return False
        
    last_6 = get_last_6_without_ties(outcomes, four_start)
    if len(last_6) < 6:
        return False
        
    # Count consecutive occurrences of the bias outcome
    target = 'P' if bias == 'PPP' else 'B'
    max_consecutive = 0
    current_consecutive = 0
    
    for outcome in last_6:
        if outcome == target:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    return max_consecutive >= 3

def get_last_6_without_ties(outcomes, four_start):
    """
    Get the last 6 Player (P) or Banker (B) outcomes, ignoring ties (T).
    
    Args:
        outcomes: List of outcomes ('P', 'B', 'T')
        four_start: Whether this started with 4 of the same outcome
        
    Returns:
        list: Last 6 non-tie outcomes
        
    Raises:
        ValueError: If outcomes contains invalid values
    """
    if any(o not in ['P', 'B', 'T'] for o in outcomes):
        raise ValueError("outcomes must only contain 'P', 'B', or 'T'")
        
    non_tie_outcomes = [o for o in outcomes if o in ['P', 'B']]
    return non_tie_outcomes[-6:]  # Only return last 6 non-tie outcomes

def get_table_bias(outcomes, bias):
    """
    Determine the table bias based on the outcomes.
    
    Args:
        outcomes: List of outcomes ('P', 'B', 'T')
        bias: Current bias ('PPP' or 'BBB')
        
    Returns:
        str: 'P' or 'B' indicating which side to bet on
        
    Raises:
        ValueError: If bias is invalid or outcomes contains invalid values
    """
    if bias not in ["PPP", "BBB"]:
        raise ValueError("bias must be 'PPP' or 'BBB'")
        
    if any(o not in ['P', 'B', 'T'] for o in outcomes):
        raise ValueError("outcomes must only contain 'P', 'B', or 'T'")
        
    # Filter out ties and get counts
    filtered_outcomes = [o for o in outcomes if o in ['P', 'B']]
    if not filtered_outcomes:
        return 'B'  # Default to Banker when no valid outcomes
        
    number_of_p = filtered_outcomes.count('P')
    number_of_b = filtered_outcomes.count('B')
    total_hands = len(filtered_outcomes)
    
    # Calculate bias percentage with minimum threshold
    min_threshold = 0.55  # Require at least 55% bias
    
    if bias == 'PPP':
        p_percentage = number_of_p / total_hands if total_hands > 0 else 0
        if p_percentage >= min_threshold:
            return 'B'  # Bet against strong player bias
        else:
            return 'P'  # No strong bias, stick with original
    else:  # BBB
        b_percentage = number_of_b / total_hands if total_hands > 0 else 0
        if b_percentage >= min_threshold:
            return 'P'  # Bet against strong banker bias
        else:
            return 'B'  # No strong bias, stick with original

def simulate_strategies(selected_strategies=None, bet_size=50, use_optimized_params=True, override_params=None):
    """
    Run simulations for selected strategies on historical data with the option to 
    use optimized parameter sets.
    
    Args:
        selected_strategies (list, optional): List of BettingStrategy enum values to run.
            If None, all strategies with parameters will be run.
        bet_size (int, optional): Flat bet size to use for all strategies
        use_optimized_params (bool, optional): If True, use optimized parameter sets
            for each strategy. If False, use default parameters.
        override_params (dict, optional): Override parameters for specific strategies.
            Keys are BettingStrategy enums, values are parameter dictionaries.
            
    Returns:
        tuple: (summary_df, all_results) - DataFrame with summary metrics and raw results dict
    """
    # Start timing
    start_time = datetime.now()
    logger.info(f"Starting simulation at {start_time}")
    
    # Read the CSV file with historical data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame(), {}
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return pd.DataFrame(), {}
    
    # Verify required columns exist
    required_columns = ['timestamp', 'initial_mode', 'all_outcomes_first_shoe']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"CSV file missing required columns: {missing_columns}")
        return pd.DataFrame(), {}
    
    # Get strategy parameters - either optimized or defaults
    if override_params and isinstance(override_params, dict):
        strategy_params = override_params
    else:
        strategy_params = get_optimized_strategy_parameters() if use_optimized_params else {}
    
    # Determine which strategies to run
    if selected_strategies:
        # Filter to only selected strategies that have parameters if needed
        if use_optimized_params:
            available_strategies = list(strategy_params.keys())
            strategies_to_run = [s for s in selected_strategies if s in available_strategies]
            if not strategies_to_run:
                logger.error(f"None of the selected strategies have optimized parameters. Available strategies: {[s.value for s in available_strategies]}")
                return pd.DataFrame(), {}
        else:
            strategies_to_run = selected_strategies
    else:
        if use_optimized_params:
            # Run all strategies with optimized parameters
            strategies_to_run = list(strategy_params.keys())
        else:
            # Run all strategies 
            strategies_to_run = list(BettingStrategy)
    
    logger.info(f"Running simulation for strategies: {[s.value for s in strategies_to_run]} with {'optimized' if use_optimized_params else 'default'} parameters")
    
    # Initialize results storage
    all_results = {strategy: [] for strategy in strategies_to_run}
    processed_lines = 0
    
    # Process each line
    for index, row in df.iterrows():
        try:
            # Extract outcomes and initial mode
            outcomes = list(row['all_outcomes_first_shoe'])
            initial_mode = row['initial_mode']
            four_start = row.get('four_start', False)
            
            # Determine starting point based on ties in first 6 outcomes
            start_from = get_initial_outcomes_count(outcomes)
            
            # Process each strategy with its parameters
            for strategy in strategies_to_run:
                if use_optimized_params and strategy in strategy_params:
                    params_list = strategy_params[strategy]
                else:
                    # Use empty dict for default parameters
                    params_list = [{}]
                
                for params in params_list:
                    try:
                        # Add description if not present
                        #if use_optimized_params and 'description' not in params:
                        #    params['description'] = 'No description'
                            
                        # Add bet sizing parameters
                        print(params)
                        bet_params = params.copy() if params else {}
                        # Override with flat bet size
                        bet_params['bet_size'] = bet_size
                        
                        # Initialize simulator with betting parameters
                        simulator = GameSimulator([], initial_mode, strategy=strategy, strategy_params=bet_params)
                        
                        # Run simulation for this line and parameter set
                        result = _simulate_single_line(
                            simulator=simulator, 
                            outcomes=outcomes, 
                            initial_mode=initial_mode, 
                            start_from=start_from, 
                            four_start=four_start,
                            strategy=strategy,
                            bet_size=bet_size,
                            params=bet_params,
                            timestamp=row['timestamp']
                        )
                        
                        if result:  # Only append if we got results
                            all_results[strategy].append(result)
                            
                    except Exception as e:
                        logger.error(f"Error simulating strategy {strategy.value} with params {params}: {e}")
                        logger.debug(traceback.format_exc())
            
            processed_lines += 1
            if processed_lines % 50 == 0:
                logger.info(f"Processed {processed_lines}/{len(df)} lines ({processed_lines/len(df)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error processing line {index}: {e}")
            logger.debug(traceback.format_exc())
    
    # Generate summary results and visualizations
    summary_df = _generate_summary_dataframe(all_results, strategy_params, strategies_to_run, use_optimized_params, bet_size)
    
    # Log execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.info(f"Simulation completed in {execution_time.total_seconds():.2f} seconds")
    
    return summary_df, all_results

def _simulate_single_line(simulator, outcomes, initial_mode, start_from, four_start, strategy, bet_size, params, timestamp):
    """Run simulation for a single line with given parameters"""
    total_pnl = 0
    shoe_bets = []
    
    # Add initial outcomes without processing
    for outcome in outcomes[:start_from]:
        simulator.outcomes.append(outcome)
        simulator.processed_outcomes.append(outcome)
    
    # Track performance metrics
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0  # positive for wins, negative for losses
    max_win_streak = 0
    max_loss_streak = 0
    
    # Process remaining outcomes one by one
    for outcome in outcomes[start_from:]:
        # Get bet decision based on strategy type
        if strategy == BettingStrategy.ORIGINAL:
            next_bet, new_mode = play_mode(simulator.current_mode, simulator.outcomes, initial_mode, simulator.last_bet, four_start)
            simulator.current_mode = new_mode
        else:
            next_bet = simulator.get_strategy_bet(len(simulator.processed_outcomes))
        
        # Skip non-betting hands
        if next_bet == 'SKIP':
            simulator.outcomes.append(outcome)
            simulator.processed_outcomes.append(outcome)
            continue

        # Create and process bet with flat bet size
        current_bet = Bet(next_bet, simulator.bet_size)
        shoe_bets.append(current_bet)
        
        # Calculate result and update metrics
        if outcome == next_bet:  # Win
            profit = current_bet.set_result('W')
            consecutive_wins += 1
            consecutive_losses = 0
            current_streak = max(current_streak + 1, 1)
            max_win_streak = max(max_win_streak, current_streak)
        elif outcome == 'T':  # Tie
            profit = current_bet.set_result('T')
            # Ties don't affect streaks
        else:  # Loss
            profit = current_bet.set_result('L')
            consecutive_losses += 1
            consecutive_wins = 0
            current_streak = min(current_streak - 1, -1)
            max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        # Update max consecutive counts
        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Update total PnL
        total_pnl += profit
        simulator.last_bet = current_bet
        
        # Update simulator state
        simulator.outcomes.append(outcome)
        simulator.processed_outcomes.append(outcome)
    
    # Calculate final statistics for this shoe
    num_bets = len(shoe_bets)
    if num_bets == 0:
        return None  # Skip if no bets were made
        
    wins = sum(1 for bet in shoe_bets if bet.result == 'W')
    losses = sum(1 for bet in shoe_bets if bet.result == 'L')
    ties = sum(1 for bet in shoe_bets if bet.result == 'T')
    win_rate = wins / (num_bets - ties) if num_bets - ties > 0 else 0
    avg_profit_per_bet = total_pnl / num_bets
    risk_reward_ratio = max_loss_streak / max_win_streak if max_win_streak > 0 else float('inf')
    
    # Record results for this parameter set
    description = params.get('description', 'No description')
    result = {
        'timestamp': timestamp,
        'initial_mode': initial_mode,
        'profit': total_pnl,
        'num_bets': num_bets,
        'win_rate': win_rate,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'risk_reward_ratio': risk_reward_ratio,
        'avg_profit_per_bet': avg_profit_per_bet,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'all_outcomes_first_shoe': ','.join(outcomes),
        'bet_size': bet_size,
        'parameters': params,
        'description': description
    }
    
    return result

def _generate_summary_dataframe(all_results, strategy_params, strategies_to_run, use_optimized_params, bet_size):
    """Generate summary dataframe from simulation results"""
    # Check if we have any results
    has_results = False
    for strategy, results in all_results.items():
        if results:
            has_results = True
            break
    
    if not has_results:
        logger.error("No results were generated during simulation")
        return pd.DataFrame()
    
    # Create a summary dataframe for easy comparison
    summary_rows = []
    
    for strategy in strategies_to_run:
        if not all_results[strategy]:  # Skip if no results for this strategy
            logger.warning(f"No results for {strategy.value}. Skipping.")
            continue
        
        try:
            strategy_df = pd.DataFrame(all_results[strategy])
            logger.info(f"Created DataFrame for {strategy.value} with {len(strategy_df)} rows")
            
            if use_optimized_params and strategy in strategy_params:
                # Group results by parameters description
                for params in strategy_params[strategy]:
                    description = params.get('description', 'No description')
                    param_str = ', '.join(f"{k}={v}" for k, v in params.items() if k != 'description')
                    
                    # Find results with matching description
                    mask = strategy_df['description'] == description
                    results_df = strategy_df[mask]
                    
                    if len(results_df) == 0:
                        logger.warning(f"No results for {strategy.value} with description '{description}'. Skipping.")
                        continue
                    
                    # Calculate summary metrics
                    summary_row = _calculate_strategy_metrics(strategy, param_str, results_df, description)
                    if summary_row:
                        summary_rows.append(summary_row)
            else:
                # Just one parameter set for this strategy
                param_str = "default"
                summary_row = _calculate_strategy_metrics(strategy, param_str, strategy_df, "Default parameters")
                if summary_row:
                    summary_rows.append(summary_row)
                
        except Exception as e:
            logger.error(f"Error processing results for strategy {strategy.value}: {e}")
            logger.debug(traceback.format_exc())
    
    if not summary_rows:
        logger.error("No summary data could be generated")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by different metrics for various visualizations
    try:
        summary_df = pd.DataFrame(summary_rows)
        logger.info(f"Created summary DataFrame with {len(summary_df)} rows")
        
        # Create and save visualizations
        _create_strategy_visualizations(summary_df, strategies_to_run, all_results, bet_size, use_optimized_params)
        
    except Exception as e:
        logger.error(f"Error creating summary DataFrame or visualizations: {e}")
        logger.debug(traceback.format_exc())
        summary_df = pd.DataFrame()
    
    return summary_df

def _calculate_strategy_metrics(strategy, param_str, results_df, description=None):
    """Calculate summary metrics for a strategy with specific parameters"""
    try:
        total_profit = results_df['profit'].sum()
        avg_profit_per_line = results_df['profit'].mean()
        win_rate = results_df['win_rate'].mean() * 100
        profitable_lines = (results_df['profit'] > 0).mean() * 100
        max_drawdown = results_df['profit'].min()
        avg_consecutive_wins = results_df['max_consecutive_wins'].mean()
        avg_consecutive_losses = results_df['max_consecutive_losses'].mean()
        avg_win_streak = results_df['max_win_streak'].mean()
        avg_loss_streak = results_df['max_loss_streak'].mean()
        avg_risk_reward = results_df['risk_reward_ratio'].replace([float('inf'), float('-inf')], np.nan).mean()
        
        # Calculate risk-adjusted metrics
        profit_std = results_df['profit'].std()
        sharpe_ratio = avg_profit_per_line / profit_std if profit_std > 0 else 0
        
        sortino_ratio = 0
        negative_returns = results_df.loc[results_df['profit'] < 0, 'profit']
        if len(negative_returns) > 0:
            neg_std = negative_returns.std()
            if neg_std > 0:
                sortino_ratio = avg_profit_per_line / neg_std
        
        # Prepare summary row for this strategy+parameter combo
        summary_row = {
            'Strategy': strategy.value,
            'Parameters': param_str,
            'Description': description if description else "No description",
            'Total Profit': total_profit,
            'Avg Profit per Line': avg_profit_per_line,
            'Win Rate': win_rate,
            'Profitable Lines %': profitable_lines,
            'Max Drawdown': max_drawdown,
            'Avg Consecutive Wins': avg_consecutive_wins,
            'Avg Consecutive Losses': avg_consecutive_losses,
            'Avg Win Streak': avg_win_streak,
            'Avg Loss Streak': avg_loss_streak,
            'Avg Risk-Reward': avg_risk_reward,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Total Lines': len(results_df),
            'Results': results_df
        }
        return summary_row
    
    except Exception as e:
        logger.error(f"Error calculating metrics for {strategy.value} with parameters '{param_str}': {e}")
        logger.debug(traceback.format_exc())
        return None

def _create_strategy_visualizations(summary_df, strategies_to_run, all_results, bet_size, use_optimized_params):
    """
    Create and save visualizations for strategy performance.
    
    Args:
        summary_df (DataFrame): Summary DataFrame with strategy metrics
        strategies_to_run (list): List of strategies that were tested
        all_results (dict): Raw results for each strategy
        bet_size (int): Flat bet size used for the simulations
        use_optimized_params (bool): Whether optimized parameters were used
    """
    try:
        # Sort summary dataframes by different metrics for various visualizations
        summary_by_profit = summary_df.sort_values('Total Profit', ascending=False)
        summary_by_sharpe = summary_df.sort_values('Sharpe Ratio', ascending=False)
        summary_by_win_rate = summary_df.sort_values('Win Rate', ascending=False)
        summary_by_profitable_lines = summary_df.sort_values('Profitable Lines %', ascending=False)
        summary_by_sortino = summary_df.sort_values('Sortino Ratio', ascending=False)
        
        # Print formatted results
        param_type = "Optimized" if use_optimized_params else "Default"
        print(f"\n{param_type} Strategy Performance Summary (Sorted by Sortino Ratio):")
        print("=" * 120)
        print(f"Using flat bet size of ${bet_size} for all strategies")
        print("=" * 120)

        for i, (_, row) in enumerate(summary_by_sortino.iterrows(), 1):
            print(f"\n{i}. Strategy: {row['Strategy']} - {row['Description']}")
            print(f"   Parameters: {row['Parameters']}")
            print(f"   Total Profit: ${row['Total Profit']:.2f}")
            print(f"   Average Profit per Line: ${row['Avg Profit per Line']:.2f}")
            print(f"   Win Rate: {row['Win Rate']:.2f}%")
            print(f"   Profitable Lines: {row['Profitable Lines %']:.1f}%")
            print(f"   Max Drawdown: ${row['Max Drawdown']:.2f}")
            print(f"   Risk Metrics - Sharpe: {row['Sharpe Ratio']:.2f}, Sortino: {row['Sortino Ratio']:.2f}")
            print("-" * 100)
        
        # Create enhanced visualizations
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Box plot of profit distribution by strategy
        plt.subplot(2, 2, 1)
        plot_data = []
        plot_labels = []
        
        # Limit to top 10 strategies by profit for readability
        top_strategies = summary_by_profit.head(10)
        
        for _, row in top_strategies.iterrows():
            if 'Results' not in row or not isinstance(row['Results'], pd.DataFrame) or len(row['Results']) == 0:
                continue
                
            strategy_name = f"{row['Strategy']}\n({row['Description']})"
            results = row['Results']
            plot_data.append(results['profit'])
            plot_labels.append(strategy_name)
        
        if plot_data:
            plt.boxplot(plot_data, labels=plot_labels, vert=True)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Profit Distribution - Top 10 Strategy Variants (Flat Bet Size: ${bet_size})')
            plt.ylabel('Profit ($)')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Win rate comparison
        plt.subplot(2, 2, 2)
        top_win_rates = summary_by_win_rate.head(10)
        y_pos = np.arange(len(top_win_rates))
        plt.barh(y_pos, top_win_rates['Win Rate'])
        plt.yticks(y_pos, [f"{s}\n({d})" for s, d in zip(top_win_rates['Strategy'], top_win_rates['Description'])])
        plt.xlabel('Win Rate (%)')
        plt.title('Win Rate Comparison - Top 10')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Risk-adjusted returns (Sharpe Ratio)
        plt.subplot(2, 2, 3)
        top_sharpe = summary_by_sharpe.head(10)
        y_pos = np.arange(len(top_sharpe))
        plt.barh(y_pos, top_sharpe['Sharpe Ratio'])
        plt.yticks(y_pos, [f"{s}\n({d})" for s, d in zip(top_sharpe['Strategy'], top_sharpe['Description'])])
        plt.xlabel('Sharpe Ratio')
        plt.title('Risk-Adjusted Returns (Sharpe Ratio) - Top 10')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Strategy Category Comparison
        plt.subplot(2, 2, 4)
        
        # Group by strategy category
        category_profit = {}
        for _, row in summary_df.iterrows():
            strategy = row['Strategy']
            
            # Determine category
            if 'hybrid' in strategy:
                category = 'Hybrid Strategies'
            elif strategy == 'original':
                category = 'Original Strategy'
            else:
                category = 'Statistical Strategies'
                
            if category not in category_profit:
                category_profit[category] = []
            category_profit[category].append(row['Total Profit'])
        
        # Calculate average profit per category
        categories = list(category_profit.keys())
        avg_profits = [np.mean(profits) for profits in category_profit.values()]
        
        plt.bar(categories, avg_profits, color=['blue', 'green', 'red'])
        plt.ylabel('Average Total Profit ($)')
        plt.title('Average Profit by Strategy Category')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the enhanced plot
        filename_prefix = "optimized" if use_optimized_params else "default"
        plot_path = os.path.join(os.path.dirname(__file__), f'{filename_prefix}_strategy_comparison_flat{bet_size}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Main visualizations saved to {plot_path}")
        
        # Save detailed results to CSV
        summary_csv = pd.DataFrame(summary_df)
        
        # Drop the 'Results' column as it contains DataFrames
        if 'Results' in summary_csv.columns:
            summary_csv = summary_csv.drop(columns=['Results'])
            
        csv_path = os.path.join(os.path.dirname(__file__), f'{filename_prefix}_strategy_comparison_flat{bet_size}.csv')
        summary_csv.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to: {csv_path}")
        print(f"Detailed results saved to: {csv_path}")
        
        # Create second visualization - time series
        _create_time_series_visualization(summary_df, all_results, bet_size, use_optimized_params)
        
        # Print top strategies by different metrics
        _print_top_strategies_by_metrics(summary_df, category_profit)
        
    except Exception as e:
        logger.error(f"Error creating strategy visualizations: {e}")
        logger.debug(traceback.format_exc())

def _create_time_series_visualization(summary_df, all_results, bet_size, use_optimized_params):
    """Create time series visualization showing cumulative profit over time for top strategies"""
    try:
        # Create a figure for strategy performance over time
        fig2 = plt.figure(figsize=(15, 10))
        
        # Select top 5 strategies for time series
        top_5_strategies = summary_df.sort_values('Total Profit', ascending=False).head(5)
        
        # Plot profit over bets for top 5 strategies
        for idx, (_, row) in enumerate(top_5_strategies.iterrows()):
            strategy_name = row['Strategy']
            description = row['Description']
            results = row['Results']
            
            # Aggregate profits across all lines for this strategy
            profits_by_line = results[['timestamp', 'profit']].sort_values('timestamp')
            
            # Plot cumulative profits
            cumulative_profits = profits_by_line['profit'].cumsum()
            plt.plot(range(len(cumulative_profits)), cumulative_profits, 
                    label=f"{strategy_name} ({description})", 
                    linewidth=2)
        
        plt.title(f'Cumulative Profit Over Time - Top 5 Strategies (Flat Bet Size: ${bet_size})')
        plt.xlabel('Number of Shoes')
        plt.ylabel('Cumulative Profit ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the time series plot
        filename_prefix = "optimized" if use_optimized_params else "default"
        timeseries_path = os.path.join(os.path.dirname(__file__), f'{filename_prefix}_strategy_profit_timeseries_flat{bet_size}.png')
        plt.savefig(timeseries_path)
        plt.close()
        
        logger.info(f"Time series visualization saved to {timeseries_path}")
    
    except Exception as e:
        logger.error(f"Error creating time series visualization: {e}")
        logger.debug(traceback.format_exc())

def _print_top_strategies_by_metrics(summary_df, category_profit):
    """Print top strategies by different performance metrics"""
    try:
        print("\nTop 3 Strategies by Different Metrics:")
        print("=" * 100)
        
        # Sort summary dataframes by different metrics
        summary_by_profit = summary_df.sort_values('Total Profit', ascending=False)
        summary_by_sharpe = summary_df.sort_values('Sharpe Ratio', ascending=False)
        summary_by_win_rate = summary_df.sort_values('Win Rate', ascending=False)
        
        metrics = {
            'Total Profit': ('${:.2f}', summary_by_profit),
            'Win Rate': ('{:.2f}%', summary_by_win_rate),
            'Sharpe Ratio': ('{:.2f}', summary_by_sharpe),
            'Sortino Ratio': ('{:.2f}', summary_df.sort_values('Sortino Ratio', ascending=False)),
            'Max Drawdown': ('${:.2f}', summary_df.sort_values('Max Drawdown', ascending=False)),
            'Profitable Lines %': ('{:.2f}%', summary_df.sort_values('Profitable Lines %', ascending=False))
        }
        
        for metric, (format_str, sorted_df) in metrics.items():
            print(f"\nTop 3 by {metric}:")
            print("-" * 50)
            top_3 = sorted_df.head(3)
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"{idx}. {row['Strategy']} ({row['Description']}): {format_str.format(row[metric])}")
        
        print("\nStrategy Type Comparison:")
        print("=" * 50)
        for category, profits in category_profit.items():
            avg_profit = np.mean(profits)
            max_profit = np.max(profits)
            min_profit = np.min(profits)
            print(f"{category}:")
            print(f"   Average Profit: ${avg_profit:.2f}")
            print(f"   Max Profit: ${max_profit:.2f}")
            print(f"   Min Profit: ${min_profit:.2f}")
            print(f"   Number of Variants: {len(profits)}")
            print("-" * 30)
    
    except Exception as e:
        logger.error(f"Error printing strategy metrics: {e}")
        logger.debug(traceback.format_exc())

# These are deprecated functions that are replaced by the new function above
def simulate_all_lines(selected_strategies=None, bet_size=50):
    """DEPRECATED: Use simulate_strategies() instead"""
    logger.warning("simulate_all_lines() is deprecated. Use simulate_strategies() instead")
    return simulate_strategies(selected_strategies, bet_size, use_optimized_params=False)

def simulate_with_optimized_parameters(selected_strategies=None, bet_size=50):
    """DEPRECATED: Use simulate_strategies() instead"""
    logger.warning("simulate_with_optimized_parameters() is deprecated. Use simulate_strategies() instead")
    return simulate_strategies(selected_strategies, bet_size, use_optimized_params=True)

def generate_parameter_combinations(base_params, param_ranges):
    """
    Generate combinations of parameters for testing using min/max ranges and number of steps.
    
    Args:
        base_params (dict): Base parameter set to modify
        param_ranges (dict): Dictionary mapping parameter names to ranges.
            Each range should be a dict with 'min', 'max', and 'steps' keys,
            or 'values' for explicit list of values to test.
            Example:
            {
                'window_size': {'min': 10, 'max': 30, 'steps': 5},
                'confidence_threshold': {'min': 0.5, 'max': 0.7, 'steps': 5},
                'discrete_param': {'values': [True, False]}  # For discrete values
            }
        
    Returns:
        list: List of parameter dictionaries with different combinations
    """
    import numpy as np
    import itertools
    
    # Generate values for each parameter
    param_values = {}
    for param_name, range_info in param_ranges.items():
        if 'values' in range_info:
            # Use explicit values if provided
            param_values[param_name] = range_info['values']
        else:
            # Generate evenly spaced values between min and max
            min_val = range_info['min']
            max_val = range_info['max']
            steps = range_info['steps']
            
            if isinstance(min_val, int) and isinstance(max_val, int):
                # For integer parameters
                values = np.linspace(min_val, max_val, steps, dtype=int)
            elif isinstance(min_val, float) or isinstance(max_val, float):
                # For float parameters
                values = np.linspace(min_val, max_val, steps)
            else:
                raise ValueError(f"Invalid parameter type for {param_name}")
                
            param_values[param_name] = values.tolist()
    
    # Get all parameter names and their possible values
    param_names = list(param_values.keys())
    value_lists = [param_values[name] for name in param_names]
    
    # Generate all possible combinations
    combinations = list(itertools.product(*value_lists))
    
    # Create parameter sets
    param_sets = []
    for combo in combinations:
        # Start with base parameters
        params = base_params.copy()
        
        # Update with current combination
        for name, value in zip(param_names, combo):
            params[name] = value
            
        param_sets.append(params)
    
    logger.info(f"Generated {len(param_sets)} parameter combinations")
    for param_name, values in param_values.items():
        logger.info(f"{param_name}: testing values {values}")
    
    return param_sets

def test_parameter_combinations(strategy, param_ranges, bet_size=50, base_params=None):
    """
    Test multiple parameter combinations for a strategy to find optimal settings.
    
    Args:
        strategy (BettingStrategy): Strategy to test
        param_ranges (dict): Dictionary mapping parameter names to lists of values to test
        bet_size (int, optional): Bet size to use. Defaults to 50.
        base_params (dict, optional): Base parameters to start from. If None, uses optimized parameters.
        
    Returns:
        tuple: (best_params, all_results) - Best parameter set and full results DataFrame
    """
    logger.info(f"Testing parameter combinations for {strategy.value}")
    
    # Get base parameters
    if base_params is None:
        from strategies.optimized_parameters import OPTIMIZED_PARAMETERS
        strategy_key = strategy.value.lower()
        if strategy_key in OPTIMIZED_PARAMETERS:
            base_params = OPTIMIZED_PARAMETERS[strategy_key].copy()
        else:
            base_params = {}
    
    # Generate parameter combinations
    param_sets = generate_parameter_combinations(base_params, param_ranges)
    logger.info(f"Generated {len(param_sets)} parameter combinations to test")
    
    # Store results for each parameter set
    results = []
    time.sleep(2)  # To avoid overwhelming the logger
    # Run simulation for each parameter set
    for param_set in param_sets:
        try:
            param_str = ", ".join(f"{k}={v}" for k, v in param_ranges.items())
            logger.info(f"Testing parameters: {param_str}")
            
            # Run simulation with these parameters
            summary_df, raw_results = simulate_strategies(
                selected_strategies=[strategy],
                bet_size=bet_size,
                use_optimized_params=True,
                override_params={strategy: param_set}
            )
            
            if len(summary_df) > 0:
                # Extract metrics
                metrics = summary_df.iloc[0].to_dict()
                
                # Add parameter values to results
                result = {
                    **{k: v for k, v in param_set.items() if k in param_ranges},
                    'Total Profit': metrics['Total Profit'],
                    'Win Rate': metrics['Win Rate'],
                    'Sharpe Ratio': metrics['Sharpe Ratio'],
                    'Sortino Ratio': metrics['Sortino Ratio'],
                    'Max Drawdown': metrics['Max Drawdown'],
                    'Parameters': param_set
                }
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error testing parameter set: {e}")
            logger.debug(traceback.format_exc())
    
    # Convert results to DataFrame
    if not results:
        logger.error("No valid results generated from parameter testing")
        return None, None
        
    results_df = pd.DataFrame(results)
    
    # Find best parameter set based on different metrics
    metrics = ['Total Profit', 'Sharpe Ratio', 'Sortino Ratio']
    best_params = {}
    
    for metric in metrics:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_params[metric] = results_df.loc[best_idx, 'Parameters']
    
    # Create visualization of parameter impacts
    _visualize_parameter_impacts(results_df, strategy, param_ranges)
    
    return best_params, results_df

def _visualize_parameter_impacts(results_df, strategy, param_ranges):
    """
    Create visualizations showing how different parameters impact strategy performance.
    
    Args:
        results_df (DataFrame): Results from parameter testing
        strategy (BettingStrategy): Strategy that was tested
        param_ranges (dict): Dictionary of parameters that were varied
    """
    try:
        num_params = len(param_ranges)
        if num_params == 0:
            return
            
        # Create subplots for each parameter
        fig = plt.figure(figsize=(15, 5 * ((num_params + 1) // 2)))
        
        for idx, (param_name, values) in enumerate(param_ranges.items(), 1):
            plt.subplot(((num_params + 1) // 2), 2, idx)
            
            # Group by parameter value and calculate mean performance
            grouped = results_df.groupby(param_name).agg({
                'Total Profit': 'mean',
                'Sharpe Ratio': 'mean',
                'Sortino Ratio': 'mean'
            }).reset_index()
            
            # Plot metrics
            x = grouped[param_name]
            plt.plot(x, grouped['Total Profit'], label='Total Profit', marker='o')
            plt.plot(x, grouped['Sharpe Ratio'] * 1000, label='Sharpe Ratio (x1000)', marker='s')
            plt.plot(x, grouped['Sortino Ratio'] * 1000, label='Sortino Ratio (x1000)', marker='^')
            
            plt.title(f'Impact of {param_name} on Performance')
            plt.xlabel(param_name)
            plt.ylabel('Metric Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), f'parameter_impact_{strategy.value}.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Parameter impact visualization saved to {plot_path}")
        
        # Save detailed results to CSV
        csv_path = os.path.join(os.path.dirname(__file__), f'parameter_testing_{strategy.value}.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Parameter testing results saved to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error creating parameter impact visualization: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    # Test a selection of optimized strategies with flat betting
    optimized_strategies = [
        BettingStrategy.ORIGINAL,
        BettingStrategy.ADAPTIVE_BIAS,
        BettingStrategy.HYBRID_ADAPTIVE,
        BettingStrategy.HYBRID_ENSEMBLE,
        BettingStrategy.ENHANCED_ADAPTIVE_BIAS,
        BettingStrategy.DYNAMIC_SKIP,
        BettingStrategy.SELECTIVE_BETTING,
        BettingStrategy.STREAK_REVERSAL_SAFE_EXIT,
        BettingStrategy.CONFIDENCE_THRESHOLD_ESCALATOR,
        BettingStrategy.REINFORCEMENT_LEARNING,
        BettingStrategy.BAYESIAN_INFERENCE
    ]
    
    # Run with flat betting size
    bet_size = 1
    print(f"\nRunning simulation with flat bet size of ${bet_size}...")
    try:
        # Use the unified simulation function
        summary_df, results = simulate_strategies(
            selected_strategies=optimized_strategies, 
            bet_size=bet_size,
            use_optimized_params=True
        )
        
        if len(summary_df) == 0:
            print("No results were generated. Check the log for details.")
        else:
            print(f"Simulation completed successfully with {len(summary_df)} strategy variants.")
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        print(f"An error occurred: {e}")

    # Example of testing parameter combinations for a strategy
    strategy = BettingStrategy.ADAPTIVE_BIAS
    
    # Define parameter ranges to test using min/max/steps format
    param_ranges = {
        'window_size': {'min': 10, 'max': 30, 'steps': 5},  # Will test 5 evenly spaced values
        'weight_recent': {'min': 1.5, 'max': 3.0, 'steps': 4},
        'confidence_threshold': {'min': 0.52, 'max': 0.64, 'steps': 5},
        'min_samples': {'min': 10, 'max': 25, 'steps': 4},
        'skip_enabled': {'values': [True, False]}  # Discrete values using 'values' key
    }
    
    print(f"\nTesting parameter combinations for {strategy.value}...")
    best_params, results = test_parameter_combinations(
        strategy=strategy,
        param_ranges=param_ranges,
        bet_size=1  # Use small bet size for testing
    )
    
    if best_params:
        print("\nBest parameters found:")
        for metric, params in best_params.items():
            print(f"\nOptimized for {metric}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    else:
        print("No valid parameter combinations found. Check the logs for details.")
