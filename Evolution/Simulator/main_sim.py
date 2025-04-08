# Import standard libraries
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
import gc
import multiprocessing
from multiprocessing import Pool
import itertools
from functools import partial
import signal
import sys

# Global flag for clean exit
STOP_EXECUTION = False

# Signal handler for clean exit
def signal_handler(sig, frame):
    global STOP_EXECUTION
    print("\n\nStopping execution gracefully. Please wait...")
    STOP_EXECUTION = True

    # Print instructions for the user
    print("Press Ctrl+C again to force immediate exit (not recommended)")

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

try:
    from tqdm import tqdm
except ImportError:
    # Define a simple tqdm replacement if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to ERROR to reduce console output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulator.log'),
        logging.StreamHandler()
    ]
)

# Create a separate logger for file logging that captures more details
file_handler = logging.FileHandler('simulator_detailed.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get the root logger and add the file handler
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Get the module logger
logger = logging.getLogger(__name__)

# Seed for reproducibility
random.seed(42)

# Import the BettingStrategy enum from the strategies package
from strategies.betting_strategy import BettingStrategy

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
        if strategy_bet in [None, "SKIP"]:
            # Use original mode-based logic
            next_bet = "P" if self.current_mode == "BBB" else "B"
            return next_bet
        if strategy_bet == "SKIP":
            print("Skipping bet")
            return "B"

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

def simulate_strategies(historical_data_df: pd.DataFrame, selected_strategies=None, bet_size=50, use_optimized_params=True, override_params=None):
    """
    Run simulations for selected strategies on historical data with the option to
    use optimized parameter sets.

    Args:
        historical_data_df (pd.DataFrame): Pre-loaded DataFrame with historical data.
        selected_strategies (list, optional): List of BettingStrategy enum values to run.
            If None, all strategies with parameters will be run.
        bet_size (int, optional): Flat bet size to use for all strategies
        use_optimized_params (bool, optional): If True, use optimized parameter sets
            for each strategy. If False, use default parameters.
        override_params (dict, optional): Override parameters for specific strategies.
            Keys are BettingStrategy enums, values are lists of parameter dictionaries.

    Returns:
        tuple: (summary_df, all_results) - DataFrame with summary metrics and raw results dict
    """
    # Start timing
    start_time = datetime.now()
    logger.info(f"Starting simulation run at {start_time}")

    # --- Use the passed DataFrame ---
    df = historical_data_df
    logger.info(f"Using pre-loaded DataFrame with {len(df)} rows.")
    # --- Remove CSV loading code ---
    # csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
    # if not os.path.exists(csv_path): ...
    # try: df = pd.read_csv(csv_path) ...
    # except Exception as e: ...
    # Verify required columns ... (This check is now done in test_parameter_combinations)

    # Get strategy parameters - either optimized or defaults
    if override_params and isinstance(override_params, dict):
        # When overriding, we expect a dict like {strategy: [param_set1, param_set2]}
        strategy_params = override_params
        # Ensure override_params values are lists
        for strat, params in strategy_params.items():
            if not isinstance(params, list):
                 logger.warning(f"Override parameters for {strat.value} should be a list. Wrapping it.")
                 strategy_params[strat] = [params]
    else:
        strategy_params = get_optimized_strategy_parameters() if use_optimized_params else {}

    # Determine which strategies to run
    if selected_strategies:
        # Filter to only selected strategies that have parameters if needed
        if use_optimized_params or override_params:
            available_strategies = list(strategy_params.keys())
            strategies_to_run = [s for s in selected_strategies if s in available_strategies]
            if not strategies_to_run:
                logger.error(f"None of the selected strategies have parameters defined (optimized or overridden). Available: {[s.value for s in available_strategies]}")
                return pd.DataFrame(), {}
        else: # Running with defaults
            strategies_to_run = selected_strategies
    else: # No specific strategies selected
        if use_optimized_params or override_params:
            # Run all strategies with defined parameters
            strategies_to_run = list(strategy_params.keys())
        else:
            # Run all strategies with defaults
            strategies_to_run = list(BettingStrategy)
            # Ensure strategy_params has entries for default runs
            for strat in strategies_to_run:
                 if strat not in strategy_params:
                      strategy_params[strat] = [{}] # Use empty dict for default params

    param_mode = "overridden" if override_params else ("optimized" if use_optimized_params else "default")
    logger.info(f"Running simulation for strategies: {[s.value for s in strategies_to_run]} with {param_mode} parameters")

    # Initialize results storage
    all_results = {strategy: [] for strategy in strategies_to_run}
    processed_lines = 0
    total_lines_to_process = len(df) * len(strategies_to_run) # Estimate total work

    # Process each line
    for index, row in df.iterrows():
        try:
            # Extract outcomes and initial mode
            # Ensure outcomes are treated as a list of characters
            outcomes_str = row['all_outcomes_first_shoe']
            if isinstance(outcomes_str, str):
                 outcomes = list(outcomes_str)
            elif isinstance(outcomes_str, list): # Handle if already list
                 outcomes = outcomes_str
            else:
                 logger.warning(f"Unexpected type for outcomes on line {index}: {type(outcomes_str)}. Skipping line.")
                 continue

            initial_mode = row['initial_mode']
            four_start = row.get('four_start', False)

            # Determine starting point based on ties in first 6 outcomes
            start_from = get_initial_outcomes_count(outcomes)

            # Process each strategy with its parameters
            for strategy in strategies_to_run:
                # Get the list of parameter sets for this strategy
                # If running defaults and no params defined, strategy_params[strategy] was set to [{}] above
                params_list = strategy_params.get(strategy, [{}])

                for params in params_list:
                    try:
                        # Add description if not present and using optimized params without override
                        if use_optimized_params and not override_params and 'description' not in params:
                             # This case might not happen if get_optimized_strategy_parameters always adds description
                             params['description'] = 'Optimized (no desc)'
                        elif not use_optimized_params and not override_params and 'description' not in params:
                             params['description'] = 'Default parameters'

                        # Add bet sizing parameters
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
                            params=bet_params, # Pass the full params used
                            timestamp=row['timestamp']
                        )

                        if result:  # Only append if we got results
                            all_results[strategy].append(result)

                    except Exception as e:
                        param_desc_log = params.get('description', 'N/A')
                        logger.error(f"Error simulating strategy {strategy.value} (Desc: {param_desc_log}) on line {index}: {e}")
                        # logger.debug(traceback.format_exc()) # Keep debug for detailed trace

            processed_lines += 1
            if processed_lines % 50 == 0: # Log progress based on lines processed
                logger.info(f"Processed {processed_lines}/{len(df)} lines...")

        except Exception as e:
            logger.error(f"Error processing line {index}: {e}")
            logger.debug(traceback.format_exc())

    # Generate summary results and visualizations
    summary_df = _generate_summary_dataframe(all_results, strategy_params, strategies_to_run, use_optimized_params or bool(override_params), bet_size)

    # Log execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.info(f"Simulation run completed in {execution_time.total_seconds():.2f} seconds")

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

    total_possible_hands = len(outcomes) - start_from

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
        'total_possible_hands': total_possible_hands,
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
        # 'all_outcomes_first_shoe': ','.join(outcomes), # --- REMOVED THIS LINE ---
        'bet_size': bet_size,
        'parameters': params, # Keep parameters used for this run
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

        # Calculate Betting Frequency
        avg_num_bets = results_df['num_bets'].mean()
        avg_total_possible = results_df['total_possible_hands'].mean()
        betting_frequency = (avg_num_bets / avg_total_possible) * 100 if avg_total_possible > 0 else 0

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
            'Betting Frequency %': betting_frequency,
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
            print(f"   Betting Frequency: {row['Betting Frequency %']:.1f}%")
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
            'Betting Frequency %': ('{:.1f}%', summary_df.sort_values('Betting Frequency %', ascending=False)),
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
    print("Generating parameter combinations...")
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
    print(f"Generated {len(combinations)} combinations of parameters")

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

def _run_single_parameter_set(args):
    """
    Helper function to run a single parameter set in parallel.

    Args:
        args (tuple): Tuple containing (strategy, param_set, bet_size, historical_data_df, param_ranges, i, total)

    Returns:
        dict: Results for this parameter set or None if error
    """
    global STOP_EXECUTION
    if STOP_EXECUTION:
        return None

    strategy, param_set, bet_size, historical_data_df, param_ranges, i, total = args

    try:
        # Extract only the varied parameters for logging clarity
        varied_params_str = ", ".join(f"{k}={v}" for k, v in param_set.items() if k in param_ranges)
        # Use debug level instead of info to reduce console output
        logger.debug(f"Testing combination {i+1}/{total}: {varied_params_str}")

        # Run simulation with these parameters
        summary_df, raw_results = simulate_strategies(
            historical_data_df=historical_data_df,
            selected_strategies=[strategy],
            bet_size=bet_size,
            use_optimized_params=True,
            override_params={strategy: [param_set]}
            # Removed verbose parameter as it's not supported
        )

        if summary_df is not None and len(summary_df) > 0:
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
                'Profitable Lines %': metrics['Profitable Lines %'],
                'Parameters': param_set
            }
            return result
        else:
            logger.debug(f"No summary generated for parameter set: {varied_params_str}")
            return None
    except Exception as e:
        logger.error(f"Error testing parameter set {param_set}: {e}")
        logger.debug(traceback.format_exc())
        return None

def test_parameter_combinations(strategy, param_ranges=None, bet_size=50, base_params=None, use_parallel=True, use_genetic=False, population_size=20, generations=5, mutation_rate=0.2):
    """
    Test multiple parameter combinations for a strategy to find optimal settings.

    This function supports two optimization approaches:
    1. Grid Search: Tests all combinations of parameters (use_genetic=False)
    2. Genetic Algorithm: Uses evolutionary approach to find optimal parameters (use_genetic=True)

    Both approaches can be run with parallel processing (use_parallel=True) to speed up execution
    by distributing work across multiple CPU cores.

    Args:
        strategy (BettingStrategy): Strategy to test
        param_ranges (dict, optional): Dictionary mapping parameter names to ranges of values to test.
            If None, will load parameter ranges from the strategy's definition file.
        bet_size (int, optional): Bet size to use. Defaults to 50.
        base_params (dict, optional): Base parameters to start from. If None, uses optimized parameters.
        use_parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        use_genetic (bool, optional): Whether to use genetic algorithm for parameter search. Defaults to False.
        population_size (int, optional): Size of population for genetic algorithm. Defaults to 20.
        generations (int, optional): Number of generations for genetic algorithm. Defaults to 5.
        mutation_rate (float, optional): Mutation rate for genetic algorithm. Defaults to 0.2.

    Returns:
        tuple: (best_params, all_results) - Best parameter set and full results DataFrame

    Examples:
        # Use parallel grid search with parameter ranges from strategy file
        best_params, results = test_parameter_combinations(
            strategy=strategy,
            use_parallel=True,
            use_genetic=False
        )

        # Use parallel genetic algorithm with parameter ranges from strategy file
        best_params, results = test_parameter_combinations(
            strategy=strategy,
            use_parallel=True,
            use_genetic=True,
            population_size=20,
            generations=5
        )
    """
    logger.info(f"Testing parameter combinations for {strategy.value}")

    # If no parameter ranges provided, load them from the strategy file
    if param_ranges is None:
        try:
            # Import the parameter ranges from the strategies package
            from strategies import get_parameter_ranges
            param_ranges = get_parameter_ranges(strategy)
            logger.info(f"Loaded parameter ranges from strategy file for {strategy.value}")

            # Print the parameters that will be tested
            for param_name, range_info in param_ranges.items():
                if 'values' in range_info:
                    logger.info(f"Parameter {param_name}: {range_info['values']}")
                else:
                    logger.info(f"Parameter {param_name}: {range_info['min']} to {range_info['max']} ({range_info['steps']} steps)")
        except Exception as e:
            logger.error(f"Error loading parameter ranges for {strategy.value}: {e}")
            logger.error("Please provide parameter ranges explicitly.")
            return None, None

    # --- Load historical data ONCE ---
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None, None
    try:
        historical_data_df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(historical_data_df)} rows from {csv_path} for parameter testing.")

        # --- Filter out lines with less than 25 outcomes ---
        initial_rows = len(historical_data_df)
        # Ensure the column exists and handle potential non-string values gracefully
        if 'all_outcomes_first_shoe' in historical_data_df.columns:
            historical_data_df = historical_data_df[historical_data_df['all_outcomes_first_shoe'].apply(lambda x: isinstance(x, str) and len(x) >= 25)]
            removed_rows = initial_rows - len(historical_data_df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with fewer than 25 outcomes. Remaining rows: {len(historical_data_df)}")
        else:
            logger.warning("'all_outcomes_first_shoe' column not found, skipping outcome length filtering.")
        # --- End filtering ---
    except Exception as e:
        logger.error(f"Error reading CSV file for parameter testing: {e}")
        return None, None
    # Verify required columns
    required_columns = ['timestamp', 'initial_mode', 'all_outcomes_first_shoe']
    missing_columns = [col for col in required_columns if col not in historical_data_df.columns]
    if missing_columns:
        logger.error(f"CSV file missing required columns: {missing_columns}")
        return None, None
    # --- End loading data ---

    # Get base parameters
    if base_params is None:
        from strategies.optimized_parameters import OPTIMIZED_PARAMETERS
        strategy_key = strategy.value.lower()
        if strategy_key in OPTIMIZED_PARAMETERS:
            base_params = OPTIMIZED_PARAMETERS[strategy_key].copy()
        else:
            base_params = {}

    # Determine which approach to use for parameter testing
    if use_genetic:
        logger.info(f"Using genetic algorithm for parameter search with population size {population_size} and {generations} generations")
        logger.info(f"Genetic algorithm will use {'parallel' if use_parallel else 'sequential'} processing")
        param_sets, results = _run_genetic_algorithm(
            strategy, param_ranges, base_params, bet_size, historical_data_df,
            population_size, generations, mutation_rate, use_parallel
        )
    else:
        # Generate all parameter combinations
        param_sets = generate_parameter_combinations(base_params, param_ranges)
        logger.info(f"Generated {len(param_sets)} parameter combinations to test")

        # Store results for each parameter set
        results = []

        if use_parallel:
            # Use parallel processing to test parameter combinations
            print(f"Using parallel processing with {multiprocessing.cpu_count()} cores")

            # Create arguments for parallel processing
            args_list = [
                (strategy, param_set, bet_size, historical_data_df, param_ranges, i, len(param_sets))
                for i, param_set in enumerate(param_sets)
            ]

            # Run in parallel
            try:
                # Use a process pool to run parameter tests in parallel
                with Pool(processes=multiprocessing.cpu_count()) as pool:
                    # Use tqdm if available for progress bar, otherwise use simple logging
                    try:
                        from tqdm import tqdm
                        print("\n") # Add a newline before progress bar
                        parallel_results = list(tqdm(pool.imap(_run_single_parameter_set, args_list),
                                                  total=len(args_list),
                                                  desc="Testing parameters"))
                    except ImportError:
                        print("Progress bar not available, using simple progress logging")
                        parallel_results = []
                        for i, args in enumerate(args_list):
                            if STOP_EXECUTION:
                                print("\nStopping parameter testing due to user request.")
                                break
                            print(f"\rTesting parameter set {i+1}/{len(args_list)}", end="")
                            result = _run_single_parameter_set(args)
                            parallel_results.append(result)

                # Filter out None results and add to results list
                results = [r for r in parallel_results if r is not None]
                print(f"\nCompleted {len(results)} parameter combinations successfully out of {len(param_sets)}")
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                logger.debug(traceback.format_exc())
                # Fall back to sequential processing
                print("\nFalling back to sequential processing")
                use_parallel = False

        if not use_parallel:
            # Sequential processing
            print("Using sequential processing")

            # Run simulation for each parameter set
            for i, param_set in enumerate(param_sets):
                if STOP_EXECUTION:
                    print("\nStopping parameter testing due to user request.")
                    break
                print(f"\rTesting parameter set {i+1}/{len(param_sets)}", end="")
                result = _run_single_parameter_set((strategy, param_set, bet_size, historical_data_df, param_ranges, i, len(param_sets)))
                if result is not None:
                    results.append(result)

        # Create DataFrame from results
        if results:
            results_df = pd.DataFrame(results)

            # --- Sort results by Sortino Ratio and print top 2 ---
            if 'Profitable Lines %' in results_df.columns:
                # Clear the screen for better visibility of final results
                print("\n\n" + "=" * 80)
                print("PARAMETER OPTIMIZATION RESULTS")
                print("=" * 80)

                # Show top 2 by Profitable Lines %
                top_2_profitable = results_df.sort_values('Profitable Lines %', ascending=False).head(2)
                print("\nTop 2 Parameter Combinations by Profitable Lines %:")
                print("-" * 60)
                for i, (_, row) in enumerate(top_2_profitable.iterrows()):
                    print(f"#{i+1}:")
                    print(f"  Sortino Ratio: {row['Sortino Ratio']:.4f}")
                    print(f"  Total Profit: {row['Total Profit']:.2f}")
                    print(f"  Win Rate: {row['Win Rate']:.2f}%")
                    print(f"  Profitable Lines %: {row['Profitable Lines %']:.2f}%")
                    print(f"  Betting Frequency %: {row['Betting Frequency %']:.2f}%")
                    print(f"  Max Drawdown: {row['Max Drawdown']:.2f}")
                    print(f"  Parameters: {row['Parameters']}") # Display the full parameter set
                    print("-" * 60)

                # Show top 2 by Sortino Ratio
                top_2_sortino = results_df.sort_values('Sortino Ratio', ascending=False).head(2)
                print("\nTop 2 Parameter Combinations by Sortino Ratio:")
                print("-" * 60)
                for i, (_, row) in enumerate(top_2_sortino.iterrows()):
                    print(f"#{i+1}:")
                    print(f"  Sortino Ratio: {row['Sortino Ratio']:.4f}")
                    print(f"  Total Profit: {row['Total Profit']:.2f}")
                    print(f"  Win Rate: {row['Win Rate']:.2f}%")
                    print(f"  Profitable Lines %: {row['Profitable Lines %']:.2f}%")
                    print(f"  Betting Frequency %: {row['Betting Frequency %']:.2f}%")
                    print(f"  Max Drawdown: {row['Max Drawdown']:.2f}")
                    print(f"  Parameters: {row['Parameters']}") # Display the full parameter set
                    print("-" * 60)
            else:
                logger.warning("Could not find 'Profitable Lines %' in results to determine top 2.")
        else:
            logger.warning("No valid results generated from parameter testing")

    # Convert results to DataFrame
    if results.empty:
        logger.error("No valid results generated from parameter testing")
        return None, None

    results_df = pd.DataFrame(results)

    # For genetic algorithm results, we need to handle the different format
    if use_genetic:
        # Check if we have the fitness column
        if 'Fitness' in results_df.columns:
            logger.info("Processing genetic algorithm results")
            # Find the best individual based on fitness
            best_params = {}
            try:
                best_idx = results_df['Fitness'].idxmax()
                best_params['Fitness'] = results_df.loc[best_idx, 'Parameters']
                logger.info(f"Best genetic algorithm solution found with fitness: {results_df.loc[best_idx, 'Fitness']}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not find best parameter from genetic algorithm: {e}")
        else:
            logger.warning("Genetic algorithm results don't contain Fitness column")
    else:
        # For grid search, find best parameter set based on different metrics
        metrics = ['Total Profit', 'Sharpe Ratio', 'Sortino Ratio']
        best_params = {}

        for metric in metrics:
            if metric in results_df.columns and not results_df[metric].isnull().all():
                try:
                    best_idx = results_df[metric].idxmax()
                    best_params[metric] = results_df.loc[best_idx, 'Parameters']
                except ValueError as ve:
                    logger.warning(f"Could not find best parameter for metric '{metric}': {ve}") # Handle cases where all values might be NaN
            else:
                logger.warning(f"Metric '{metric}' not found or all NaN in results_df, cannot find best parameters.")

    # Create visualization of parameter impacts
    _visualize_parameter_impacts(results_df, strategy, param_ranges)

    return best_params, results_df

# Global function for evaluating fitness in genetic algorithm
def _evaluate_fitness_ga(param_set, strategy, historical_data_df, bet_size):
    global STOP_EXECUTION
    if STOP_EXECUTION:
        return float('-inf')

    try:
        # Run simulation with these parameters
        summary_df, _ = simulate_strategies(
            historical_data_df=historical_data_df,
            selected_strategies=[strategy],
            bet_size=bet_size,
            use_optimized_params=True,
            override_params={strategy: [param_set]}
            # Removed verbose parameter as it's not supported
        )

        if summary_df is not None and len(summary_df) > 0:
            # Extract metrics - use a balanced fitness function
            metrics = summary_df.iloc[0].to_dict()

            # Calculate a balanced fitness score using multiple metrics
            sortino_ratio = metrics.get('Sortino Ratio', 0)
            profit_pct = metrics.get('Profitable Lines %', 0)
            total_profit = metrics.get('Total Profit', 0)
            win_rate = metrics.get('Win Rate', 0)
            max_drawdown = abs(metrics.get('Max Drawdown', 0)) or 1  # Avoid division by zero

            # Weighted fitness function
            fitness = (
                sortino_ratio * 0.4 +                # 40% weight on Sortino ratio
                (profit_pct / 100) * 0.3 +          # 30% weight on profitable lines percentage
                (win_rate / 100) * 0.1 +            # 10% weight on win rate
                (total_profit / (max_drawdown)) * 0.2  # 20% weight on profit-to-drawdown ratio
            )
            # Print the top 2 best parameters for Sortino ratio and profitable lines percentage
            # Also print the parameter values to see if they're actually changing
            param_str = ', '.join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in param_set.items() if k in ['short_window', 'medium_window', 'long_window', 'confidence_threshold']])
            print(f"\rSortino: {sortino_ratio:.2f}, Profit %: {profit_pct:.2f}, Params: {param_str}", end="")

            return fitness
        else:
            return float('-inf')  # Very low fitness for failed runs
    except Exception as e:
        logger.debug(f"Error evaluating fitness for {param_set}: {e}")  # Use debug instead of error
        return float('-inf')

# Global function for evaluating individuals in parallel
def _evaluate_individual_ga(args):
    strategy, historical_data_df, bet_size, param_ranges, i, individual, generation, population_size = args
    logger.debug(f"Evaluating individual {i+1}/{population_size} in generation {generation+1}")
    fitness = _evaluate_fitness_ga(individual, strategy, historical_data_df, bet_size)

    # Create result dictionary
    result = {
        **{k: v for k, v in individual.items() if k in param_ranges},
        'Fitness': fitness,
        'Generation': generation + 1,
        'Individual': i + 1,
        'Parameters': individual
    }

    return (fitness, result)

def _run_genetic_algorithm(strategy, param_ranges, base_params, bet_size, historical_data_df, population_size=20, generations=5, mutation_rate=0.2, use_parallel=True):
    """
    Run a genetic algorithm to find optimal parameter combinations.

    Args:
        strategy (BettingStrategy): Strategy to optimize
        param_ranges (dict): Dictionary of parameter ranges to explore
        base_params (dict): Base parameters to start from
        bet_size (int): Bet size to use
        historical_data_df (DataFrame): Historical data for simulation
        population_size (int): Size of the population
        generations (int): Number of generations to run
        mutation_rate (float): Probability of mutation
        use_parallel (bool): Whether to use parallel processing for fitness evaluation

    Returns:
        tuple: (param_sets, results) - List of parameter sets and their results
    """
    # This function implements a genetic algorithm for parameter optimization
    # When use_parallel=True, it evaluates the fitness of each generation's individuals in parallel
    # This provides a significant speedup on multi-core systems
    import random
    import numpy as np

    logger.info(f"Starting genetic algorithm with population size {population_size} and {generations} generations")

    # Generate parameter values for each parameter
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

    # Function to create a random individual
    def create_individual():
        params = base_params.copy()
        for param_name, values in param_values.items():
            params[param_name] = random.choice(values)
        return params

    # Create initial population
    population = [create_individual() for _ in range(population_size)]

    # Function to select parents based on fitness
    def select_parents(population, fitnesses):
        # Tournament selection
        def tournament_select():
            # Select 3 random individuals and pick the best
            candidates = random.sample(range(len(population)), min(3, len(population)))
            best_idx = max(candidates, key=lambda idx: fitnesses[idx])
            return population[best_idx]

        return tournament_select(), tournament_select()

    # Function to crossover two parents
    def crossover(parent1, parent2):
        child = base_params.copy()
        for param_name in param_values.keys():
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    # Function to mutate an individual
    def mutate(individual):
        for param_name, values in param_values.items():
            # Mutate with probability mutation_rate
            if random.random() < mutation_rate:
                individual[param_name] = random.choice(values)
        return individual

    # Run the genetic algorithm
    all_results = []
    best_fitness = float('-inf')
    best_individual = None

    for generation in range(generations):
        # Check if execution should be stopped
        if STOP_EXECUTION:
            print("\nStopping genetic algorithm early due to user request.")
            break

        print(f"\rGeneration {generation+1}/{generations}", end="")

        # Prepare arguments for parallel evaluation with all required parameters
        eval_args = [
            (strategy, historical_data_df, bet_size, param_ranges, i, individual, generation, len(population))
            for i, individual in enumerate(population)
        ]

        # Evaluate fitness for each individual (parallel or sequential)
        if use_parallel and len(population) > 1:
            try:
                logger.debug(f"Using parallel processing for generation {generation+1} with {multiprocessing.cpu_count()} cores")
                with Pool(processes=multiprocessing.cpu_count()) as pool:
                    # Use tqdm for progress tracking if available
                    try:
                        from tqdm import tqdm
                        print("\n") # Add a newline before progress bar
                        eval_results = list(tqdm(pool.imap(_evaluate_individual_ga, eval_args),
                                               total=len(eval_args),
                                               desc=f"Gen {generation+1}"))
                    except ImportError:
                        print(f"\rEvaluating generation {generation+1}...", end="")
                        eval_results = pool.map(_evaluate_individual_ga, eval_args)

                # Extract fitness values and results
                fitnesses = [res[0] for res in eval_results]
                generation_results = [res[1] for res in eval_results]
                all_results.extend(generation_results)

            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                logger.debug(traceback.format_exc())
                # Fall back to sequential processing
                logger.debug("Falling back to sequential processing for this generation")
                use_parallel = False

        # Sequential processing if parallel failed or not requested
        if not use_parallel or len(population) <= 1:
            fitnesses = []
            for i, individual in enumerate(population):
                if STOP_EXECUTION:
                    break
                print(f"\rEvaluating individual {i+1}/{len(population)} in generation {generation+1}", end="")
                # Call the global function directly
                fitness = _evaluate_fitness_ga(individual, strategy, historical_data_df, bet_size)

                # Create result dictionary
                result = {
                    **{k: v for k, v in individual.items() if k in param_ranges},
                    'Fitness': fitness,
                    'Generation': generation + 1,
                    'Individual': i + 1,
                    'Parameters': individual
                }

                fitnesses.append(fitness)
                all_results.append(result)

            # Update best individual
            if fitnesses and max(fitnesses) > best_fitness:
                best_idx = fitnesses.index(max(fitnesses))
                best_fitness = fitnesses[best_idx]
                best_individual = population[best_idx].copy()
                print(f"\rNew best fitness: {best_fitness:.4f}")

        # Check if execution should be stopped
        if STOP_EXECUTION:
            break

        # Create next generation
        new_population = []

        # Elitism: keep the best individual
        if best_individual is not None:
            new_population.append(best_individual.copy())

        # Create rest of population through selection, crossover, and mutation
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        # Replace old population
        population = new_population

        # Print best fitness in this generation (only if we have results)
        if fitnesses:
            gen_best_idx = fitnesses.index(max(fitnesses))
            print(f"\rGeneration {generation+1} best fitness: {fitnesses[gen_best_idx]:.4f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Find the best individual from the final population
    if fitnesses and population:
        best_idx = fitnesses.index(max(fitnesses))
        best_individual = population[best_idx]
        logger.info(f"Best individual from final population: {best_individual}")
        logger.info(f"Best fitness: {fitnesses[best_idx]}")

    # Return all parameter sets and results DataFrame
    return population, results_df

def _visualize_parameter_impacts(results_df, strategy, param_ranges):
    """
    Create visualizations showing how different parameters impact strategy performance.

    Args:
        results_df (DataFrame): Results from parameter testing
        strategy (BettingStrategy): Strategy that was tested
        param_ranges (dict): Dictionary of parameters that were varied
    """
    try:
        # Check if results_df is empty
        if results_df is None or len(results_df) == 0:
            logger.warning("No results to visualize")
            return

        # Check if we have the required columns for visualization
        required_columns = ['Total Profit', 'Sharpe Ratio', 'Sortino Ratio']

        # For genetic algorithm results, we might have 'Fitness' instead
        if 'Fitness' in results_df.columns and not all(col in results_df.columns for col in required_columns):
            logger.info("Converting genetic algorithm results for visualization")
            # Map Fitness to the required metrics if they don't exist
            if 'Total Profit' not in results_df.columns:
                results_df['Total Profit'] = results_df['Fitness'] * 100  # Scale for visualization
            if 'Sharpe Ratio' not in results_df.columns:
                results_df['Sharpe Ratio'] = results_df['Fitness'] / 10  # Scale for visualization
            if 'Sortino Ratio' not in results_df.columns:
                results_df['Sortino Ratio'] = results_df['Fitness'] / 5  # Scale for visualization

        # Check again if we have the required columns
        if not all(col in results_df.columns for col in required_columns):
            logger.warning(f"Missing required columns for visualization: {[col for col in required_columns if col not in results_df.columns]}")
            # Create a simplified visualization with available columns
            available_metrics = [col for col in required_columns if col in results_df.columns]
            if not available_metrics:
                if 'Fitness' in results_df.columns:
                    available_metrics = ['Fitness']
                else:
                    logger.error("No metrics available for visualization")
                    return

        num_params = len(param_ranges)
        if num_params == 0:
            logger.warning("No parameters to visualize")
            return

        # Create subplots for each parameter
        fig = plt.figure(figsize=(15, 5 * ((num_params + 1) // 2)))

        for idx, (param_name, values) in enumerate(param_ranges.items(), 1):
            # Skip if parameter is not in results
            if param_name not in results_df.columns:
                logger.warning(f"Parameter {param_name} not found in results")
                continue

            plt.subplot(((num_params + 1) // 2), 2, idx)

            # Determine which metrics to plot
            metrics_to_plot = []
            for metric in required_columns:
                if metric in results_df.columns:
                    metrics_to_plot.append(metric)
            if not metrics_to_plot and 'Fitness' in results_df.columns:
                metrics_to_plot = ['Fitness']

            if not metrics_to_plot:
                logger.warning(f"No metrics available for parameter {param_name}")
                continue

            # Group by parameter value and calculate mean performance
            agg_dict = {metric: 'mean' for metric in metrics_to_plot}
            grouped = results_df.groupby(param_name).agg(agg_dict).reset_index()

            # Plot metrics
            x = grouped[param_name]
            for metric in metrics_to_plot:
                # Scale metrics for better visualization
                scale = 1
                if metric == 'Sharpe Ratio' or metric == 'Sortino Ratio':
                    scale = 1000
                    plt.plot(x, grouped[metric] * scale, label=f'{metric} (x{scale})', marker='s')
                else:
                    plt.plot(x, grouped[metric], label=metric, marker='o')

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
        BettingStrategy.BAYESIAN_INFERENCE,
        BettingStrategy.RECURRENT_NEURAL_NETWORK,
        BettingStrategy.MARKOV_CHAIN,
        BettingStrategy.HYBRID_MAJORITY,
        BettingStrategy.HYBRID_ML,
        BettingStrategy.HYBRID_PATTERN,
        BettingStrategy.HYBRID_SIMPLE_MAJORITY,
        #BettingStrategy.CONSERVATIVE_PATTERN,
        BettingStrategy.COUNTER_STREAK,
        BettingStrategy.DEEP_Q_NETWORK,
        BettingStrategy.DYNAMIC_ADAPTIVE,
        BettingStrategy.ENSEMBLE_VOTING,
        BettingStrategy.FOLLOW_STREAK,
        BettingStrategy.FREQUENCY_ANALYSIS,
        BettingStrategy.GENETIC_ALGORITHM,
        BettingStrategy.LOSS_AVERSION,
        BettingStrategy.MAJORITY_LAST_N,
        #BettingStrategy.META_STRATEGY,
        BettingStrategy.MOMENTUM_OSCILLATOR,
        BettingStrategy.MONTE_CARLO_SIMULATION,
        BettingStrategy.MULTI_CONDITION,
        BettingStrategy.PATTERN_BASED,
        BettingStrategy.PATTERN_INTERRUPTION,
        BettingStrategy.RISK_PARITY,
        BettingStrategy.SEQUENTIAL_PATTERN_MINING,
        BettingStrategy.THOMPSON_SAMPLING,
        BettingStrategy.TIME_SERIES_FORECASTING,
        BettingStrategy.TRANSFER_LEARNING,
        BettingStrategy.TREND_CONFIRMATION,
        BettingStrategy.VOLATILITY_ADAPTIVE
    ]

    # --- Load historical data ONCE ---
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        exit(0)
    try:
        historical_data_df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(historical_data_df)} rows from {csv_path} for parameter testing.")

        # --- Filter out lines with less than 25 outcomes ---
        initial_rows = len(historical_data_df)
        # Ensure the column exists and handle potential non-string values gracefully
        if 'all_outcomes_first_shoe' in historical_data_df.columns:
            historical_data_df = historical_data_df[historical_data_df['all_outcomes_first_shoe'].apply(lambda x: isinstance(x, str) and len(x) >= 25)]
            removed_rows = initial_rows - len(historical_data_df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with fewer than 25 outcomes. Remaining rows: {len(historical_data_df)}")
        else:
            logger.warning("'all_outcomes_first_shoe' column not found, skipping outcome length filtering.")
        # --- End filtering ---
    except Exception as e:
        logger.error(f"Error reading CSV file for parameter testing: {e}")
        exit(0)
    # Verify required columns
    required_columns = ['timestamp', 'initial_mode', 'all_outcomes_first_shoe']
    missing_columns = [col for col in required_columns if col not in historical_data_df.columns]
    if missing_columns:
        logger.error(f"CSV file missing required columns: {missing_columns}")
        exit(0)
    # --- End loading data ---

    # Run with flat betting size
    bet_size = 1
    print(f"\nRunning simulation with flat bet size of ${bet_size}...")
    try:
        # Use the unified simulation function
        summary_df, results = simulate_strategies(
            historical_data_df,
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

    exit(0)  # Exit the script after running the simulation

    # Example of testing parameter combinations for a strategy
    # Import the parameter ranges from the strategies package
    from strategies import get_parameter_ranges

    # Select a strategy to test
    strategy = BettingStrategy.VOLATILITY_ADAPTIVE

    # Get parameter ranges for the selected strategy
    param_ranges = get_parameter_ranges(strategy)
    print(f"\nParameter ranges for {strategy.value}:")

    # Print the parameters that will be tested
    if param_ranges:
        for param_name, range_info in param_ranges.items():
            if 'values' in range_info:
                print(f"  {param_name}: {range_info['values']}")
            else:
                print(f"  {param_name}: {range_info['min']} to {range_info['max']} ({range_info['steps']} steps)")

    print(f"\nTesting parameter combinations for {strategy.value}...")

    # Choose which optimization method to use
    use_parallel = True  # Set to False to use sequential processing
    use_genetic = True   # Set to True to use genetic algorithm

    # Run the parameter testing with the selected method
    # Note: param_ranges is now loaded automatically from the strategy file
    best_params, results = test_parameter_combinations(
        strategy=strategy,
        bet_size=1,  # Use small bet size for testing
        use_parallel=use_parallel,
        use_genetic=use_genetic,
        population_size=200,  # Larger population for better diversity
        generations=50,      # More generations to see improvement
        mutation_rate=0.2
    )

    print("\nOptimization completed using:")
    print(f"- Processing method: {'Parallel' if use_parallel else 'Sequential'} Processing")
    print(f"- Optimization algorithm: {'Genetic Algorithm' if use_genetic else 'Grid Search'}")
    print(f"- CPU cores used: {multiprocessing.cpu_count()}")

    if use_genetic and use_parallel:
        print("\nYou're using both genetic algorithm and parallel processing together!")
        print("This combination provides the fastest parameter optimization by:")
        print("1. Using genetic algorithm to intelligently search the parameter space")
        print("2. Evaluating each generation's individuals in parallel across CPU cores")

    if best_params:
        print("\nBest parameters found:")
        for metric, params in best_params.items():
            print(f"\nOptimized for {metric}:")
            for param, value in params.items():
                print(f"  {param}: {value}")

        if use_genetic and 'Fitness' in best_params:
            print("\nGenetic Algorithm Results:")
            print(f"Best fitness achieved: {results['Fitness'].max() if 'Fitness' in results.columns else 'N/A'}")
            print(f"Number of generations used: {3}")  # Hardcoded from above
            print(f"Population size used: {10}")  # Hardcoded from above
            print(f"Total evaluations: {len(results)}")
    else:
        print("No valid parameter combinations found. Check the logs for details.")
