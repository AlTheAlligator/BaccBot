"""
Nameless betting system simulation for the Evolution simulator.

This module integrates the Line class with the main simulator to enable
nameless betting system simulations.
"""

import logging
import pandas as pd
import os
import traceback
import gc
import multiprocessing
from multiprocessing import Pool
import itertools
import numpy as np
import random
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from functools import partial

from strategies.betting_strategy import BettingStrategy
from line import Line

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

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Define a simple tqdm replacement if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

def simulate_nameless_strategies(
    historical_data_df: pd.DataFrame,
    selected_strategies=None,
    initial_left_cubes: List[int] = None,
    initial_right_cubes: List[int] = None,
    use_optimized_params=True,
    override_params=None,
    stop_loss: float = -4000
):
    """
    Run simulations for selected strategies using the nameless betting system.

    Args:
        historical_data_df: Pre-loaded DataFrame with historical data
        selected_strategies: List of BettingStrategy enum values to run
        initial_left_cubes: Initial left cubes for each line (default: [1])
        initial_right_cubes: Initial right cubes for each line (default: [1])
        use_optimized_params: If True, use optimized parameter sets
        override_params: Override parameters for specific strategies
        stop_loss: PnL threshold at which to stop the line (default: -4000)

    Returns:
        tuple: (summary_df, all_results) - DataFrame with summary metrics and raw results dict
    """
    # Start timing
    start_time = datetime.now()
    logger.info(f"Starting nameless simulation run at {start_time}")

    # Use the passed DataFrame, but only where there are more then 25 outcomes
    df = historical_data_df[historical_data_df['all_outcomes_first_shoe'].apply(lambda x: isinstance(x, str) and len(x) >= 25)]
    logger.info(f"Using pre-loaded DataFrame with {len(df)} rows.")

    # Set default cube values if not provided
    if initial_left_cubes is None:
        initial_left_cubes = [4,5,6,7,8,9,10,11,12,13,14]
    if initial_right_cubes is None:
        initial_right_cubes = [44,40,36,33,30,27,24,21,18,17,16,15]

    # Import strategy-related functions from main_sim
    from main_sim import get_optimized_strategy_parameters, get_initial_outcomes_count

    # Get strategy parameters - either optimized or defaults
    if override_params and isinstance(override_params, dict):
        strategy_params = override_params
        for strat, params in strategy_params.items():
            if not isinstance(params, list):
                logger.warning(f"Override parameters for {strat.value} should be a list. Wrapping it.")
                strategy_params[strat] = [params]
    else:
        strategy_params = get_optimized_strategy_parameters() if use_optimized_params else {}

    # Determine which strategies to run
    if selected_strategies:
        if use_optimized_params or override_params:
            available_strategies = list(strategy_params.keys())
            strategies_to_run = [s for s in selected_strategies if s in available_strategies]
            if not strategies_to_run:
                logger.error(f"None of the selected strategies have parameters defined. Available: {[s.value for s in available_strategies]}")
                return pd.DataFrame(), {}
        else:  # Running with defaults
            strategies_to_run = selected_strategies
    else:  # No specific strategies selected
        if use_optimized_params or override_params:
            strategies_to_run = list(strategy_params.keys())
        else:
            strategies_to_run = list(BettingStrategy)
            for strat in strategies_to_run:
                if strat not in strategy_params:
                    strategy_params[strat] = [{}]  # Use empty dict for default params

    param_mode = "overridden" if override_params else ("optimized" if use_optimized_params else "default")
    logger.info(f"Running nameless simulation for strategies: {[s.value for s in strategies_to_run]} with {param_mode} parameters")

    # Initialize results storage
    all_results = {strategy: [] for strategy in strategies_to_run}
    processed_lines = 0

    # Process each line
    for index, row in df.iterrows():
        try:
            # Extract outcomes and initial mode
            outcomes_str = row['all_outcomes_first_shoe']
            if isinstance(outcomes_str, str):
                outcomes = list(outcomes_str)
            elif isinstance(outcomes_str, list):
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
                params_list = strategy_params.get(strategy, [{}])

                for params in params_list:
                    try:
                        # Add description if not present
                        if use_optimized_params and not override_params and 'description' not in params:
                            params['description'] = 'Optimized (no desc)'
                        elif not use_optimized_params and not override_params and 'description' not in params:
                            params['description'] = 'Default parameters'

                        # Run simulation for this line and parameter set
                        result = _simulate_nameless_line(
                            strategy=strategy,
                            outcomes=outcomes,
                            initial_mode=initial_mode,
                            start_from=start_from,
                            four_start=four_start,
                            params=params,
                            left_cubes=initial_left_cubes,
                            right_cubes=initial_right_cubes,
                            timestamp=row['timestamp'],
                            stop_loss=stop_loss
                        )

                        if result:  # Only append if we got results
                            all_results[strategy].append(result)

                    except Exception as e:
                        param_desc_log = params.get('description', 'N/A')
                        logger.error(f"Error simulating strategy {strategy.value} (Desc: {param_desc_log}) on line {index}: {e}")
                        logger.debug(traceback.format_exc())

            processed_lines += 1
            if processed_lines % 50 == 0:
                logger.info(f"Processed {processed_lines}/{len(df)} lines...")

        except Exception as e:
            logger.error(f"Error processing line {index}: {e}")
            logger.debug(traceback.format_exc())

    # Generate summary DataFrame
    summary_df = _generate_nameless_summary_dataframe(all_results, strategies_to_run)

    # Log completion
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Nameless simulation completed in {duration}. Processed {processed_lines} lines.")
    logger.info(f"Generated {len(summary_df)} strategy variants.")

    return summary_df, all_results


def _simulate_nameless_line(
    strategy: BettingStrategy,
    outcomes: List[str],
    initial_mode: str,
    start_from: int,
    four_start: bool,  # Kept for compatibility with main_sim but not used
    params: Dict[str, Any],
    left_cubes: List[int],
    right_cubes: List[int],
    timestamp: str,
    stop_loss: float = -4000
) -> Optional[Dict[str, Any]]:
    """
    Run nameless betting system simulation for a single line with given parameters.

    Args:
        strategy: Strategy enum value to use
        outcomes: List of outcomes ('P', 'B', 'T')
        initial_mode: Initial mode ('PPP' or 'BBB')
        start_from: Index to start processing from
        four_start: Whether to use four-start logic
        params: Strategy parameters
        left_cubes: Initial left cubes
        right_cubes: Initial right cubes
        timestamp: Timestamp of the line
        stop_loss: PnL threshold at which to stop the line (default: -4000)

    Returns:
        dict: Simulation results or None if no bets were made
    """
    # Import necessary classes and functions from main_sim
    from main_sim import GameSimulator

    # Initialize simulator with betting parameters
    simulator = GameSimulator([], initial_mode, strategy=strategy, strategy_params=params)

    # Add initial outcomes without processing
    for outcome in outcomes[:start_from]:
        simulator.outcomes.append(outcome)
        simulator.processed_outcomes.append(outcome)

    # Create a line for nameless betting
    initial_side = 'B' if initial_mode == 'PPP' else 'P'  # Opposite of initial mode
    line = Line(initial_side, left_cubes.copy(), right_cubes.copy(), stop_loss)

    # Track performance metrics
    total_bets = 0
    wins = 0
    losses = 0
    ties = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    consecutive_wins = 0
    consecutive_losses = 0

    # Process remaining outcomes one by one
    for outcome in outcomes[start_from:]:
        # Skip if line is no longer active
        if not line.is_active:
            break

        # Get bet decision based on strategy
        next_bet = simulator.get_strategy_bet(len(simulator.processed_outcomes))

        # Skip non-betting hands
        if next_bet == 'SKIP':
            next_bet = initial_side
            #simulator.outcomes.append(outcome)
            #simulator.processed_outcomes.append(outcome)
            #continue

        # Place bet using the nameless system
        bet = line.place_bet(next_bet)
        if bet is None:
            # Line is no longer active
            simulator.outcomes.append(outcome)
            simulator.processed_outcomes.append(outcome)
            continue

        # Process the outcome
        result = line.process_outcome(outcome)
        #print(f"Outcome: {outcome}, Result: {bet.result}, Profit: {bet.profit:.2f}, Total PnL: {line.pnl:.2f}")

        # Update simulator state
        simulator.outcomes.append(outcome)
        simulator.processed_outcomes.append(outcome)

        # Update metrics
        total_bets += 1

        if bet.result == 'W':
            wins += 1
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        elif bet.result == 'L':
            losses += 1
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        elif bet.result == 'T':
            ties += 1
            # Ties don't affect consecutive counts

    # Skip if no bets were made
    if total_bets == 0:
        return None

    # Calculate final statistics
    win_rate = wins / (wins + losses) * 100 if wins + losses > 0 else 0
    profit = line.pnl

    # Calculate Sharpe and Sortino ratios if we have enough data
    if total_bets >= 10:
        # Extract profits from each bet
        bet_profits = [bet.profit for bet in line.bets]

        # Calculate mean and standard deviation
        mean_profit = sum(bet_profits) / len(bet_profits)
        variance = sum((p - mean_profit) ** 2 for p in bet_profits) / len(bet_profits)
        std_dev = variance ** 0.5

        # Calculate downside deviation (for Sortino)
        downside_returns = [min(0, p - mean_profit) ** 2 for p in bet_profits]
        downside_deviation = (sum(downside_returns) / len(downside_returns)) ** 0.5

        # Calculate ratios
        sharpe_ratio = mean_profit / std_dev if std_dev > 0 else 0
        sortino_ratio = mean_profit / downside_deviation if downside_deviation > 0 else 0
    else:
        sharpe_ratio = 0
        sortino_ratio = 0

    # Create result dictionary
    result = {
        'Strategy': strategy.value,
        'Description': params.get('description', 'No description'),
        'Parameters': params,
        'Initial Mode': initial_mode,
        'Total Bets': total_bets,
        'Wins': wins,
        'Losses': losses,
        'Ties': ties,
        'Win Rate': win_rate,
        'Total Profit': profit,
        'Commission Paid': line.commission_paid,
        'Max Consecutive Wins': max_consecutive_wins,
        'Max Consecutive Losses': max_consecutive_losses,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Final Left Cubes': line.left_cubes,
        'Final Right Cubes': line.right_cubes,
        'Is Active': line.is_active,
        'Timestamp': timestamp,
        'Line': line  # Store the line object for further analysis
    }

    return result


def _generate_nameless_summary_dataframe(all_results, strategies_to_run):
    """
    Generate summary dataframe from nameless simulation results.

    Args:
        all_results: Dictionary of simulation results
        strategies_to_run: List of strategies that were run

    Returns:
        DataFrame: Summary of simulation results aggregated by strategy and parameter set
    """
    # Check if we have any results
    has_results = False
    for strategy in strategies_to_run:
        if strategy in all_results and all_results[strategy]:
            has_results = True
            break

    if not has_results:
        logger.warning("No results to summarize")
        return pd.DataFrame()

    # Group results by strategy and parameter set
    grouped_results = {}

    for strategy in strategies_to_run:
        if strategy not in all_results or not all_results[strategy]:
            continue

        for result in all_results[strategy]:
            # Create a key for this strategy and parameter set
            strategy_name = result['Strategy']
            description = result['Description']
            params_str = str(sorted([(k, v) for k, v in result['Parameters'].items() if k != 'description']))
            key = (strategy_name, description, params_str)

            # Initialize group if it doesn't exist
            if key not in grouped_results:
                grouped_results[key] = {
                    'Strategy': strategy_name,
                    'Description': description,
                    'Parameters': result['Parameters'],
                    'Total Bets': 0,
                    'Wins': 0,
                    'Losses': 0,
                    'Ties': 0,
                    'Total Profit': 0.0,
                    'Commission Paid': 0.0,
                    'Max Consecutive Wins': 0,
                    'Max Consecutive Losses': 0,
                    'Sharpe Ratio': 0.0,  # Will be recalculated
                    'Sortino Ratio': 0.0,  # Will be recalculated
                    'Betting Frequency %': 0.0,  # Will be recalculated
                    'Lines Completed': 0,
                    'Lines Active': 0,
                    'Total Lines': 0,
                    'All Profits': [],  # Store all profits for statistical calculations
                    'All Results': []  # Store all individual results
                }

            # Accumulate statistics
            group = grouped_results[key]
            group['Total Bets'] += result['Total Bets']
            group['Wins'] += result['Wins']
            group['Losses'] += result['Losses']
            group['Ties'] += result['Ties']
            group['Total Profit'] += result['Total Profit']
            group['Commission Paid'] += result['Commission Paid']
            group['Max Consecutive Wins'] = max(group['Max Consecutive Wins'], result['Max Consecutive Wins'])
            group['Max Consecutive Losses'] = max(group['Max Consecutive Losses'], result['Max Consecutive Losses'])
            group['Total Lines'] += 1

            if result['Is Active']:
                group['Lines Active'] += 1
            else:
                group['Lines Completed'] += 1

            # Store individual bet profits for statistical calculations
            if 'Line' in result and hasattr(result['Line'], 'bets'):
                for bet in result['Line'].bets:
                    group['All Profits'].append(bet.profit)

            # Store the full result
            group['All Results'].append(result)

    # Calculate derived statistics for each group
    summary_data = []

    for key, group in grouped_results.items():
        # Calculate win rate
        total_decisions = group['Wins'] + group['Losses']
        win_rate = group['Wins'] / total_decisions * 100 if total_decisions > 0 else 0

        # Calculate betting frequency
        total_outcomes = sum(len(result['Line'].outcomes) for result in group['All Results'])
        betting_frequency = group['Total Bets'] / total_outcomes * 100 if total_outcomes > 0 else 0

        # Calculate Sharpe and Sortino ratios if we have enough data
        all_profits = group['All Profits']
        if len(all_profits) >= 10:
            mean_profit = sum(all_profits) / len(all_profits)
            variance = sum((p - mean_profit) ** 2 for p in all_profits) / len(all_profits)
            std_dev = variance ** 0.5

            # Calculate downside deviation (for Sortino)
            downside_returns = [min(0, p - mean_profit) ** 2 for p in all_profits]
            downside_deviation = (sum(downside_returns) / len(downside_returns)) ** 0.5

            # Calculate ratios
            sharpe_ratio = mean_profit / std_dev if std_dev > 0 else 0
            sortino_ratio = mean_profit / downside_deviation if downside_deviation > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Create summary entry
        summary_entry = {
            'Strategy': group['Strategy'],
            'Description': group['Description'],
            'Total Bets': group['Total Bets'],
            'Win Rate': win_rate,
            'Total Profit': group['Total Profit'],
            'Commission Paid': group['Commission Paid'],
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Consecutive Wins': group['Max Consecutive Wins'],
            'Max Consecutive Losses': group['Max Consecutive Losses'],
            'Betting Frequency %': betting_frequency,
            'Lines Completed': group['Lines Completed'],
            'Lines Active': group['Lines Active'],
            'Total Lines': group['Total Lines'],
            'Completion Rate %': group['Lines Completed'] / group['Total Lines'] * 100 if group['Total Lines'] > 0 else 0,
            'Profit Per Line': group['Total Profit'] / group['Total Lines'] if group['Total Lines'] > 0 else 0,
            'Profit Per Bet': group['Total Profit'] / group['Total Bets'] if group['Total Bets'] > 0 else 0,
            'Parameters': group['Parameters'],
            'All Results': group['All Results']  # Store all individual results for detailed analysis
        }

        summary_data.append(summary_entry)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Sort by Total Profit (descending)
    if not summary_df.empty and 'Total Profit' in summary_df.columns:
        summary_df = summary_df.sort_values('Total Profit', ascending=False)

    return summary_df


def generate_parameter_combinations(base_params, param_ranges):
    """
    Generate combinations of parameters for testing using min/max ranges and number of steps.
    Supports nested parameter structures.

    Args:
        base_params (dict): Base parameter set to modify
        param_ranges (dict): Dictionary mapping parameter names to ranges.
            Each range should be a dict with 'min', 'max', and 'steps' keys,
            or 'values' for explicit list of values to test,
            or a nested dictionary of parameter ranges for nested parameters.
            Example:
            {
                'window_size': {'min': 10, 'max': 30, 'steps': 5},
                'confidence_threshold': {'min': 0.5, 'max': 0.7, 'steps': 5},
                'discrete_param': {'values': [True, False]},  # For discrete values
                'nested_params': {  # For nested parameters
                    'param1': {'min': 1, 'max': 5, 'steps': 5},
                    'param2': {'values': [True, False]}
                }
            }

    Returns:
        list: List of parameter dictionaries with different combinations
    """
    print("Generating parameter combinations...")

    def _is_range_dict(d):
        """Check if a dictionary is a parameter range specification."""
        return 'values' in d or ('min' in d and 'max' in d and 'steps' in d) or 'special' in d

    def _generate_values_for_range(range_info):
        """Generate values for a single parameter range."""
        if 'values' in range_info:
            # Use explicit values if provided
            return range_info['values']
        elif 'special' in range_info and range_info['special'] == 'oscillator_phases':
            # Special handling for oscillator phases
            # We'll return a placeholder here and handle it specially later
            return ['__oscillator_phases__']
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
                raise ValueError(f"Invalid parameter type with min={min_val}, max={max_val}")

            return values.tolist()

    def _flatten_param_ranges(param_ranges, prefix='', result=None):
        """Flatten nested parameter ranges into a flat dictionary."""
        if result is None:
            result = {}

        for key, value in param_ranges.items():
            full_key = f"{prefix}{key}" if prefix else key

            if _is_range_dict(value):
                # This is a parameter range specification
                result[full_key] = value
            elif isinstance(value, dict):
                # This is a nested dictionary of parameters
                _flatten_param_ranges(value, f"{full_key}.", result)
            else:
                raise ValueError(f"Invalid parameter range specification for {full_key}")

        return result

    # Flatten the parameter ranges
    flat_param_ranges = _flatten_param_ranges(param_ranges)

    # Generate values for each parameter
    param_values = {}
    for param_name, range_info in flat_param_ranges.items():
        param_values[param_name] = _generate_values_for_range(range_info)

    # Get all parameter names and their possible values
    param_names = list(param_values.keys())
    value_lists = [param_values[name] for name in param_names]

    # Check if we have any special oscillator_phases parameters
    has_oscillator_phases = any(values == ['__oscillator_phases__'] for values in value_lists)
    has_num_oscillators = 'num_oscillators' in param_ranges

    # If we have oscillator_phases but no num_oscillators, we need to add it
    if has_oscillator_phases and not has_num_oscillators:
        logger.warning("Found 'phases' parameter but no 'num_oscillators' parameter. Using default num_oscillators=5.")
        # Add a default num_oscillators parameter
        if 'num_oscillators' not in base_params:
            base_params['num_oscillators'] = 5

    # Generate all possible combinations
    combinations = list(itertools.product(*value_lists))
    print(f"Generated {len(combinations)} combinations of parameters")

    # Create parameter sets
    param_sets = []
    for combo in combinations:
        # Start with base parameters
        params = base_params.copy() if base_params else {}

        # First pass: process all parameters except oscillator_phases
        for name, value in zip(param_names, combo):
            # Skip oscillator_phases for now
            if value == '__oscillator_phases__':
                continue

            # Handle nested parameters using dot notation
            if '.' in name:
                parts = name.split('.')
                current = params

                # Navigate to the correct nested level
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value at the deepest level
                current[parts[-1]] = value
            else:
                # This is a top-level parameter
                params[name] = value

        # Second pass: handle oscillator_phases specially
        for name, value in zip(param_names, combo):
            if value == '__oscillator_phases__':
                # Get the number of oscillators from the params
                num_oscillators = params.get('num_oscillators', 5)  # Default to 5 if not set

                # Generate random phases between 0 and 2*pi
                phases = np.random.uniform(0, 2*np.pi, num_oscillators).tolist()

                # Set the phases parameter
                if '.' in name:
                    parts = name.split('.')
                    current = params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = phases
                else:
                    params[name] = phases

        param_sets.append(params)

    logger.info(f"Generated {len(param_sets)} parameter combinations")
    for param_name, values in param_values.items():
        if values == ['__oscillator_phases__']:
            logger.info(f"{param_name}: generating random phases for each parameter set")
        else:
            logger.info(f"{param_name}: testing values {values}")

    return param_sets


def _run_single_parameter_set(args):
    """
    Helper function to run a single parameter set in parallel.

    Args:
        args (tuple): Tuple containing (strategy, param_set, historical_data_df, left_cubes, right_cubes, stop_loss, param_ranges, i, total)

    Returns:
        dict: Results for this parameter set or None if error
    """
    global STOP_EXECUTION
    if STOP_EXECUTION:
        return None

    strategy, param_set, historical_data_df, left_cubes, right_cubes, stop_loss, param_ranges, i, total = args

    try:
        # Extract only the varied parameters for logging clarity
        varied_params_str = ", ".join(f"{k}={v}" for k, v in param_set.items() if k in param_ranges)
        # Use debug level instead of info to reduce console output
        logger.debug(f"Testing combination {i+1}/{total}: {varied_params_str}")

        # Run simulation with these parameters
        summary_df, _ = simulate_nameless_strategies(
            historical_data_df=historical_data_df,
            selected_strategies=[strategy],
            initial_left_cubes=left_cubes,
            initial_right_cubes=right_cubes,
            use_optimized_params=True,
            override_params={strategy: [param_set]},
            stop_loss=stop_loss
        )

        # Extract results for this strategy
        if summary_df is not None and not summary_df.empty:
            # Get the first row (should be only one for this strategy and parameter set)
            result_row = summary_df.iloc[0].to_dict()

            # Add parameter information
            result_row['Parameters'] = param_set

            # Add varied parameters as separate columns for easier analysis
            for param_name, param_value in param_set.items():
                if param_name in param_ranges:
                    result_row[param_name] = param_value

            return result_row
        else:
            logger.warning(f"No results for parameter set {i+1}/{total}")
            return None
    except Exception as e:
        logger.error(f"Error testing parameter set {i+1}/{total}: {e}")
        logger.debug(traceback.format_exc())
        return None


# Function for evaluating fitness in genetic algorithm
def _calculate_fitness(metrics_dict, fitness_function, sortino_ratio, profit_pct, total_profit, win_rate, max_drawdown, losing_lines, total_lines, profitable_lines_pct, betting_frequency_pct, avg_games_per_line):
    """
    Calculate fitness based on the selected fitness function.

    Args:
        metrics_dict: Dictionary of all metrics
        fitness_function: The fitness function to use
        sortino_ratio: Sortino ratio
        profit_pct: Completion rate percentage
        total_profit: Total profit
        win_rate: Win rate
        max_drawdown: Maximum drawdown
        losing_lines: Number of losing lines
        total_lines: Total number of lines
        profitable_lines_pct: Percentage of profitable lines
        betting_frequency_pct: Betting frequency percentage
        avg_games_per_line: Average games per line

    Returns:
        float: Fitness score
    """
    # Calculate fitness based on the selected fitness function
    if fitness_function == 'balanced':
        # Balanced approach (default)
        fitness = (
            sortino_ratio * 0.4 +                # 40% weight on Sortino ratio
            (profit_pct / 100) * 0.3 +          # 30% weight on completion rate
            (win_rate / 100) * 0.1 +            # 10% weight on win rate
            (total_profit / 1000) * 0.2         # 20% weight on total profit (normalized)
        )

    elif fitness_function == 'robust':
        # Focus on consistency and risk management
        # Heavily penalize drawdowns and losing lines
        max_dd_penalty = min(1.0, max_drawdown / 2000)  # Normalize max drawdown with cap at 1.0
        losing_lines_pct = (losing_lines / total_lines * 100) if total_lines > 0 else 100

        fitness = (
            sortino_ratio * 0.35 +                # 35% weight on Sortino ratio
            (profitable_lines_pct / 100) * 0.25 + # 25% weight on profitable lines
            (1 - max_dd_penalty) * 0.25 +       # 25% weight on avoiding large drawdowns
            (1 - (losing_lines_pct / 100)) * 0.15 # 15% weight on avoiding losing lines
        )

    elif fitness_function == 'profit_focused':
        # Focus on maximizing profit
        # This is more aggressive and may lead to more overfitting
        fitness = (
            (total_profit / 2000) * 0.5 +        # 50% weight on total profit (normalized)
            sortino_ratio * 0.3 +                # 30% weight on Sortino ratio
            (win_rate / 100) * 0.2              # 20% weight on win rate
        )

    elif fitness_function == 'consistency':
        # Focus on consistency of returns
        # Prioritize strategies that win consistently with lower variance
        betting_consistency = betting_frequency_pct / 100  # How consistently the strategy places bets
        avg_games_consistency = min(1.0, avg_games_per_line / 50)  # Normalize with cap at 1.0

        fitness = (
            sortino_ratio * 0.4 +                # 40% weight on Sortino ratio
            (profitable_lines_pct / 100) * 0.3 + # 30% weight on profitable lines
            betting_consistency * 0.15 +         # 15% weight on betting consistency
            avg_games_consistency * 0.15         # 15% weight on consistent game length
        )

    elif fitness_function == 'anti_overfitting':
        # Specifically designed to combat overfitting
        # Penalizes complex strategies and extreme parameter values

        # We don't have access to parameter ranges here, so we use a simpler approach
        # that penalizes strategies with very high or very low betting frequency
        betting_extremity = abs(betting_frequency_pct - 50) / 50  # 0 = balanced, 1 = extreme

        # Penalize strategies with very few or very many games per line
        games_extremity = 0
        if avg_games_per_line < 10:
            games_extremity = (10 - avg_games_per_line) / 10  # Penalize very short lines
        elif avg_games_per_line > 50:
            games_extremity = (avg_games_per_line - 50) / 50  # Penalize very long lines (cap at 1.0)
            games_extremity = min(1.0, games_extremity)

        # Calculate overall extremity penalty (0 to 1)
        extremity_penalty = (betting_extremity + games_extremity) / 2

        fitness = (
            sortino_ratio * 0.3 +                # 30% weight on Sortino ratio
            (profitable_lines_pct / 100) * 0.3 + # 30% weight on profitable lines
            (1 - extremity_penalty) * 0.4        # 40% weight on avoiding extreme strategies
        )
    elif fitness_function == 'alligator_fitness':
        # Balanced approach (default)
        fitness = (
            sortino_ratio * 0.3 +                # 40% weight on Sortino ratio
            (win_rate / 100) * 0.5 +            # 50% weight on win rate
            (total_profit / 1000) * 0.2         # 20% weight on total profit (normalized)
        )
    else:
        # Default to balanced approach if unknown fitness function
        fitness = (
            sortino_ratio * 0.4 +                # 40% weight on Sortino ratio
            (profit_pct / 100) * 0.3 +          # 30% weight on completion rate
            (win_rate / 100) * 0.1 +            # 10% weight on win rate
            (total_profit / 1000) * 0.2         # 20% weight on total profit (normalized)
        )

    return fitness


def _calculate_additional_metrics(strategy, all_results):
    """
    Calculate additional metrics from simulation results.

    Args:
        strategy (BettingStrategy): Strategy that was simulated
        all_results (dict): Dictionary of simulation results

    Returns:
        dict: Dictionary of additional metrics
    """
    # Initialize metrics
    avg_games_per_line = 0
    max_drawdown = 0  # Highest drawdown across all lines (active or closed)
    max_closed_drawdown = 0  # Highest drawdown where the line closed
    avg_drawdown = 0
    drawdown_count = 0
    losing_lines = 0
    profitable_lines = 0
    total_lines = 0
    total_bets = 0
    total_possible_bets = 0
    total_games = 0

    # Extract detailed metrics from all_results if available
    if strategy in all_results and all_results[strategy]:
        total_lines = len(all_results[strategy])

        for result in all_results[strategy]:
            # Calculate games per line
            if 'Line' in result and hasattr(result['Line'], 'outcomes'):
                outcomes_count = len(result['Line'].outcomes)
                total_games += outcomes_count
                # Count total possible bets (excluding first 7 games where we don't bet)
                total_possible_bets += max(0, outcomes_count - 7)

            # Calculate drawdown metrics and check for losing/profitable lines
            if 'Line' in result and hasattr(result['Line'], 'bets'):
                # Count actual bets made
                total_bets += len(result['Line'].bets)

                # Calculate drawdown for this line
                running_pnl = 0
                peak_pnl = 0
                current_drawdown = 0
                max_line_drawdown = 0

                for bet in result['Line'].bets:
                    running_pnl += bet.profit
                    if running_pnl > peak_pnl:
                        peak_pnl = running_pnl
                    else:
                        current_drawdown = peak_pnl - running_pnl
                        max_line_drawdown = max(max_line_drawdown, current_drawdown)

                # Update max drawdown across all lines
                max_drawdown = max(max_drawdown, max_line_drawdown)

                # Update max closed drawdown if this line is no longer active
                is_line_closed = not result.get('Is Active', True)  # Default to True if not specified
                if is_line_closed:
                    max_closed_drawdown = max(max_closed_drawdown, max_line_drawdown)

                # Only count lines with actual drawdown
                if max_line_drawdown > 0:
                    avg_drawdown += max_line_drawdown
                    drawdown_count += 1

                # Check if this is a losing line (final PnL is negative)
                if running_pnl < 0:
                    losing_lines += 1
                # Check if this is a profitable line (final PnL is positive)
                elif running_pnl > 0:
                    profitable_lines += 1

        # Calculate average games per line
        if total_lines > 0:
            avg_games_per_line = total_games / total_lines

        # Calculate average drawdown on lines with drawdown
        if drawdown_count > 0:
            avg_drawdown = avg_drawdown / drawdown_count

    # Calculate profitable lines percentage and betting frequency
    profitable_lines_pct = (profitable_lines / total_lines * 100) if total_lines > 0 else 0
    betting_frequency_pct = (total_bets / total_possible_bets * 100) if total_possible_bets > 0 else 0

    # Return all calculated metrics
    return {
        'Avg Games Per Line': avg_games_per_line,
        'Max Drawdown': max_drawdown,  # Highest drawdown across all lines (active or closed)
        'Max Closed Drawdown': max_closed_drawdown,  # Highest drawdown where the line closed
        'Avg Drawdown': avg_drawdown,
        'Drawdown Count': drawdown_count,
        'Losing Lines': losing_lines,
        'Total Lines': total_lines,
        'Losing Lines %': (losing_lines / total_lines * 100) if total_lines > 0 else 0,
        'Profitable Lines %': profitable_lines_pct,
        'Betting Frequency %': betting_frequency_pct,
        'Total Bets': total_bets,
        'Total Games': total_games
    }


def _evaluate_fitness_ga(param_set, strategy, historical_data_df, left_cubes, right_cubes, stop_loss, fitness_function='balanced'):
    """
    Evaluate the fitness of a parameter set for genetic algorithm optimization.

    Args:
        param_set (dict): Parameter set to evaluate
        strategy (BettingStrategy): Strategy to test
        historical_data_df (DataFrame): Historical data for simulation
        left_cubes (list): Initial left cubes configuration
        right_cubes (list): Initial right cubes configuration
        stop_loss (float): Stop loss threshold
        fitness_function (str): The fitness function to use. Options:
            - 'balanced': Balanced approach (default)
            - 'robust': Focus on consistency and risk management
            - 'profit_focused': Focus on maximizing profit
            - 'consistency': Focus on consistency of returns
            - 'anti_overfitting': Specifically designed to combat overfitting

    Returns:
        tuple: (fitness, metrics_dict) - Fitness score and dictionary of metrics, or (-inf, {}) if error
    """
    global STOP_EXECUTION
    if STOP_EXECUTION:
        return float('-inf'), {}

    try:
        # Run simulation with these parameters
        summary_df, all_results = simulate_nameless_strategies(
            historical_data_df=historical_data_df,
            selected_strategies=[strategy],
            initial_left_cubes=left_cubes,
            initial_right_cubes=right_cubes,
            use_optimized_params=True,
            override_params={strategy: [param_set]},
            stop_loss=stop_loss
        )

        if summary_df is not None and len(summary_df) > 0:
            # Extract metrics - use a balanced fitness function
            metrics = summary_df.iloc[0].to_dict()

            # Calculate a balanced fitness score using multiple metrics
            sortino_ratio = metrics.get('Sortino Ratio', 0)
            profit_pct = metrics.get('Completion Rate %', 0)  # Using completion rate as a proxy for profitable lines
            total_profit = metrics.get('Total Profit', 0)
            win_rate = metrics.get('Win Rate', 0)

            # Calculate additional metrics using the helper function
            additional_metrics = _calculate_additional_metrics(strategy, all_results)

            # Merge the additional metrics with the basic metrics
            metrics_dict = {
                'Sortino Ratio': sortino_ratio,
                'Completion Rate %': profit_pct,
                'Total Profit': total_profit,
                'Win Rate': win_rate,
                **additional_metrics  # Add all additional metrics
            }

            # Calculate fitness using the _calculate_fitness function
            fitness = _calculate_fitness(
                metrics_dict,
                fitness_function,
                sortino_ratio,
                profit_pct,
                total_profit,
                win_rate,
                additional_metrics['Max Drawdown'],
                additional_metrics['Losing Lines'],
                additional_metrics['Total Lines'],
                additional_metrics['Profitable Lines %'],
                additional_metrics['Betting Frequency %'],
                additional_metrics['Avg Games Per Line']
            )

            # Print more detailed metrics for each evaluation
            # Only show the top 2 best parameters for clarity
            top_params = sorted([(k, v) for k, v in param_set.items() if k not in ['description']],
                                key=lambda x: str(x[0]))[:2]
            param_str = ', '.join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in top_params])

            # Only print the most important metrics to reduce console clutter
            print(f"\rSortino: {sortino_ratio:.2f}, Profit: ${total_profit:.2f}, Profitable: {additional_metrics['Profitable Lines %']:.1f}%, Betting Freq: {additional_metrics['Betting Frequency %']:.1f}%, Params: {param_str}", end="")

            return fitness, metrics_dict
        else:
            return float('-inf'), {}
    except Exception as e:
        logger.debug(f"Error evaluating fitness for {param_set}: {e}")  # Use debug instead of error
        return float('-inf'), {}


# Function for evaluating individuals in parallel
def _evaluate_individual_ga(args):
    """
    Evaluate a single individual in the genetic algorithm population.

    Args:
        args (tuple): Arguments for evaluation

    Returns:
        tuple: (fitness, result) - Fitness score and result dictionary
    """
    strategy, historical_data_df, left_cubes, right_cubes, stop_loss, param_ranges, i, individual, generation, population_size, fitness_function = args
    logger.debug(f"Evaluating individual {i+1}/{population_size} in generation {generation+1}")

    fitness, metrics_dict = _evaluate_fitness_ga(individual, strategy, historical_data_df, left_cubes, right_cubes, stop_loss, fitness_function)

    # Create result dictionary
    result = {
        **{k: v for k, v in individual.items() if k in param_ranges},
        'Fitness': fitness,
        'Generation': generation + 1,
        'Individual': i + 1,
        'Parameters': individual
    }

    # Add all metrics to the result
    if metrics_dict:
        for metric_name, metric_value in metrics_dict.items():
            result[metric_name] = metric_value

    return (fitness, result)


def _run_genetic_algorithm(strategy, param_ranges, base_params, historical_data_df, left_cubes, right_cubes, stop_loss, population_size=20, generations=5, mutation_rate=0.2, use_parallel=True, fitness_function='balanced'):
    """
    Run a genetic algorithm to find optimal parameter combinations for nameless betting system.

    Args:
        strategy (BettingStrategy): Strategy to optimize
        param_ranges (dict): Dictionary of parameter ranges to explore
        base_params (dict): Base parameters to start from
        historical_data_df (DataFrame): Historical data for simulation
        left_cubes (list): Initial left cubes configuration
        right_cubes (list): Initial right cubes configuration
        stop_loss (float): Stop loss threshold
        population_size (int): Size of the population
        generations (int): Number of generations to run
        mutation_rate (float): Probability of mutation
        use_parallel (bool): Whether to use parallel processing for fitness evaluation
        fitness_function (str): The fitness function to use. Options:
            - 'balanced': Balanced approach (default)
            - 'robust': Focus on consistency and risk management
            - 'profit_focused': Focus on maximizing profit
            - 'consistency': Focus on consistency of returns
            - 'anti_overfitting': Specifically designed to combat overfitting

    Returns:
        tuple: (param_sets, results) - List of parameter sets and their results
    """
    # This function implements a genetic algorithm for parameter optimization
    # When use_parallel=True, it evaluates the fitness of each generation's individuals in parallel
    # This provides a significant speedup on multi-core systems
    import random
    import numpy as np
    from multiprocessing import Pool, cpu_count

    logger.info(f"Starting genetic algorithm with population size {population_size} and {generations} generations")

    # Reuse the helper functions from generate_parameter_combinations
    def _is_range_dict(d):
        """Check if a dictionary is a parameter range specification."""
        return 'values' in d or ('min' in d and 'max' in d and 'steps' in d) or 'special' in d

    def _generate_values_for_range(range_info):
        """Generate values for a single parameter range."""
        if 'values' in range_info:
            # Use explicit values if provided
            return range_info['values']
        elif 'special' in range_info and range_info['special'] == 'oscillator_phases':
            # Special handling for oscillator phases
            # We'll return a placeholder here and handle it specially in create_individual
            return ['__oscillator_phases__']
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
                raise ValueError(f"Invalid parameter type with min={min_val}, max={max_val}")

            return values.tolist()

    def _flatten_param_ranges(param_ranges, prefix='', result=None):
        """Flatten nested parameter ranges into a flat dictionary."""
        if result is None:
            result = {}

        for key, value in param_ranges.items():
            full_key = f"{prefix}{key}" if prefix else key

            if _is_range_dict(value):
                # This is a parameter range specification
                result[full_key] = value
            elif isinstance(value, dict):
                # This is a nested dictionary of parameters
                _flatten_param_ranges(value, f"{full_key}.", result)
            else:
                raise ValueError(f"Invalid parameter range specification for {full_key}")

        return result

    # Flatten the parameter ranges
    flat_param_ranges = _flatten_param_ranges(param_ranges)

    # Generate values for each parameter
    param_values = {}
    for param_name, range_info in flat_param_ranges.items():
        param_values[param_name] = _generate_values_for_range(range_info)

    # Function to create a random individual
    def create_individual():
        params = base_params.copy() if base_params else {}

        # First pass: process all parameters except oscillator_phases
        # This ensures we have num_oscillators set before generating phases
        for param_name, values in param_values.items():
            # Skip oscillator_phases for now
            if values == ['__oscillator_phases__']:
                continue

            # Handle nested parameters using dot notation
            if '.' in param_name:
                parts = param_name.split('.')
                current = params

                # Navigate to the correct nested level
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value at the deepest level
                current[parts[-1]] = random.choice(values)
            else:
                # This is a top-level parameter
                params[param_name] = random.choice(values)

        # Second pass: handle oscillator_phases specially
        for param_name, values in param_values.items():
            if values == ['__oscillator_phases__']:
                # Get the number of oscillators from the params
                num_oscillators = params.get('num_oscillators', 5)  # Default to 5 if not set

                # Generate random phases between 0 and 2*pi
                phases = np.random.uniform(0, 2*np.pi, num_oscillators).tolist()

                # Set the phases parameter
                if '.' in param_name:
                    parts = param_name.split('.')
                    current = params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = phases
                else:
                    params[param_name] = phases

        return params

    # Create initial population
    population = [create_individual() for _ in range(population_size)]

    # Function to select parents based on fitness
    def select_parents(population, fitnesses):
        # Get indices of individuals sorted by fitness (descending)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

        # Select only from the top 20% of individuals
        elite_count = max(2, int(len(population) * 0.2))  # At least 2 individuals
        elite_indices = sorted_indices[:elite_count]

        # Tournament selection from the elite individuals
        def tournament_select():
            # Select 2 random individuals from the elite group and pick the best
            if len(elite_indices) <= 1:
                return population[elite_indices[0]]
            candidates = random.sample(elite_indices, min(2, len(elite_indices)))
            best_idx = max(candidates, key=lambda idx: fitnesses[idx])
            return population[best_idx]

        return tournament_select(), tournament_select()

    # Function to crossover two parents
    def crossover(parent1, parent2):
        child = base_params.copy() if base_params else {}

        # First pass: handle all parameters except oscillator_phases
        # This ensures we have num_oscillators set before handling phases
        for param_name, values in param_values.items():
            # Skip oscillator_phases for now
            if values == ['__oscillator_phases__']:
                continue

            if '.' in param_name:
                # This is a nested parameter
                parts = param_name.split('.')

                # Get parent values, handling the case where the nested structure might not exist
                parent1_value = _get_nested_value(parent1, parts)
                parent2_value = _get_nested_value(parent2, parts)

                # 50% chance to inherit from each parent if both have the value
                if parent1_value is not None and parent2_value is not None:
                    if random.random() < 0.5:
                        _set_nested_value(child, parts, parent1_value)
                    else:
                        _set_nested_value(child, parts, parent2_value)
                # Otherwise use the one that exists, or a random value if neither exists
                elif parent1_value is not None:
                    _set_nested_value(child, parts, parent1_value)
                elif parent2_value is not None:
                    _set_nested_value(child, parts, parent2_value)
                else:
                    # Neither parent has this value, use a random value from the range
                    _set_nested_value(child, parts, random.choice(values))
            else:
                # This is a top-level parameter
                # 50% chance to inherit from each parent
                if random.random() < 0.5 and param_name in parent1:
                    child[param_name] = parent1[param_name]
                elif param_name in parent2:
                    child[param_name] = parent2[param_name]
                else:
                    # Neither parent has this parameter, use a random value
                    child[param_name] = random.choice(values)

        # Second pass: handle oscillator_phases specially
        for param_name, values in param_values.items():
            if values == ['__oscillator_phases__']:
                # Get the number of oscillators from the child
                num_oscillators = child.get('num_oscillators', 5)  # Default to 5 if not set

                # Get parent phases
                if '.' in param_name:
                    parts = param_name.split('.')
                    parent1_phases = _get_nested_value(parent1, parts)
                    parent2_phases = _get_nested_value(parent2, parts)
                else:
                    parent1_phases = parent1.get(param_name, [])
                    parent2_phases = parent2.get(param_name, [])

                # Crossover the phases
                if parent1_phases and parent2_phases:
                    # If both parents have phases, do a proper crossover
                    # Adjust lengths if needed
                    if len(parent1_phases) != num_oscillators:
                        parent1_phases = _adjust_phases_length(parent1_phases, num_oscillators)
                    if len(parent2_phases) != num_oscillators:
                        parent2_phases = _adjust_phases_length(parent2_phases, num_oscillators)

                    # Create child phases by mixing parent phases
                    child_phases = []
                    for i in range(num_oscillators):
                        # 50% chance to inherit from each parent
                        if random.random() < 0.5:
                            child_phases.append(parent1_phases[i])
                        else:
                            child_phases.append(parent2_phases[i])
                elif parent1_phases:
                    # Only parent1 has phases
                    child_phases = _adjust_phases_length(parent1_phases, num_oscillators)
                elif parent2_phases:
                    # Only parent2 has phases
                    child_phases = _adjust_phases_length(parent2_phases, num_oscillators)
                else:
                    # Neither parent has phases, generate random ones
                    child_phases = np.random.uniform(0, 2*np.pi, num_oscillators).tolist()

                # Set the phases in the child
                if '.' in param_name:
                    _set_nested_value(child, parts, child_phases)
                else:
                    child[param_name] = child_phases

        return child

    # Helper function to adjust phases length
    def _adjust_phases_length(phases, target_length):
        """Adjust the length of a phases array to match the target length."""
        if len(phases) == target_length:
            return phases
        elif len(phases) > target_length:
            # Truncate
            return phases[:target_length]
        else:
            # Extend by adding random phases
            additional = np.random.uniform(0, 2*np.pi, target_length - len(phases)).tolist()
            return phases + additional

    # Helper function to get a nested value from a dictionary
    def _get_nested_value(d, keys):
        """Get a value from a nested dictionary using a list of keys."""
        current = d
        for key in keys[:-1]:
            if key not in current:
                return None
            current = current[key]
        return current.get(keys[-1]) if keys[-1] in current else None

    # Helper function to set a nested value in a dictionary
    def _set_nested_value(d, keys, value):
        """Set a value in a nested dictionary using a list of keys."""
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    # Function to mutate an individual
    def mutate(individual):
        for param_name, values in param_values.items():
            # Skip special oscillator_phases placeholder
            if values == ['__oscillator_phases__']:
                # Special handling for oscillator phases
                if random.random() < mutation_rate:
                    # Get the current phases
                    if '.' in param_name:
                        parts = param_name.split('.')
                        current_phases = _get_nested_value(individual, parts)
                    else:
                        current_phases = individual.get(param_name, [])

                    # Get the number of oscillators
                    num_oscillators = len(current_phases) if current_phases else individual.get('num_oscillators', 5)

                    # Mutate the phases - either completely new phases or modify existing ones
                    if not current_phases or random.random() < 0.3:  # 30% chance for completely new phases
                        new_phases = np.random.uniform(0, 2*np.pi, num_oscillators).tolist()
                    else:
                        # Modify existing phases by adding some noise
                        new_phases = []
                        for phase in current_phases:
                            # Add noise to the phase, keeping it within [0, 2π]
                            noise = random.uniform(-0.5, 0.5)  # Noise level
                            new_phase = (phase + noise) % (2*np.pi)
                            new_phases.append(new_phase)

                    # Set the new phases
                    if '.' in param_name:
                        _set_nested_value(individual, parts, new_phases)
                    else:
                        individual[param_name] = new_phases
                continue

            # Regular parameter mutation
            if random.random() < mutation_rate:
                if '.' in param_name:
                    # This is a nested parameter
                    parts = param_name.split('.')

                    # Mutate the nested parameter
                    _set_nested_value(individual, parts, random.choice(values))
                else:
                    # This is a top-level parameter
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
            (strategy, historical_data_df, left_cubes, right_cubes, stop_loss, param_ranges, i, individual, generation, len(population), fitness_function)
            for i, individual in enumerate(population)
        ]

        # Evaluate fitness for each individual (parallel or sequential)
        if use_parallel and len(population) > 1:
            try:
                logger.debug(f"Using parallel processing for generation {generation+1} with {cpu_count()} cores")
                with Pool(processes=cpu_count()) as pool:
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
                fitness, metrics_dict = _evaluate_fitness_ga(individual, strategy, historical_data_df, left_cubes, right_cubes, stop_loss, fitness_function)

                # Create result dictionary
                result = {
                    **{k: v for k, v in individual.items() if k in param_ranges},
                    'Fitness': fitness,
                    'Generation': generation + 1,
                    'Individual': i + 1,
                    'Parameters': individual
                }

                # Add all metrics to the result
                if metrics_dict:
                    for metric_name, metric_value in metrics_dict.items():
                        result[metric_name] = metric_value

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

        # Sort population by fitness (descending)
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

        # Helper function to check if an individual is already in the population
        def is_duplicate(individual, population):
            """Check if an individual is a duplicate of any in the population."""
            for existing in population:
                # Check if all parameter values match
                if all(individual.get(key) == existing.get(key) for key in individual.keys() if key in existing):
                    return True
            return False

        # Elitism: keep the top 20% of individuals
        elite_count = max(1, int(population_size * 0.2))  # At least 1 individual
        elite_count = min(elite_count, len(sorted_indices))

        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(population[idx].copy())

        # Create the rest of the population through selection, crossover, and mutation
        # from the top 20% of individuals
        attempts = 0
        max_attempts = population_size * 10  # Limit attempts to prevent infinite loops

        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1

            # 80% chance to create a new child through crossover and mutation
            if random.random() < 0.8:
                parent1, parent2 = select_parents(population, fitnesses)
                child = crossover(parent1, parent2)
                child = mutate(child)

                # Only add if not a duplicate
                if not is_duplicate(child, new_population):
                    new_population.append(child)
            else:
                # 20% chance to create a completely new random individual for diversity
                new_individual = create_individual()

                # Only add if not a duplicate
                if not is_duplicate(new_individual, new_population):
                    new_population.append(new_individual)

        # If we couldn't fill the population with unique individuals, fill the rest with random ones
        while len(new_population) < population_size:
            new_individual = create_individual()
            new_population.append(new_individual)
            print("\rWarning: Added random individual to maintain population size", end="")

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


def test_parameter_combinations(strategy, param_ranges=None, base_params=None, historical_data_df=None, left_cubes=None, right_cubes=None, stop_loss=-4000, use_parallel=True, use_genetic=False, population_size=20, generations=5, mutation_rate=0.2, use_cross_validation=True, validation_split=0.3, random_seed=42, fitness_function='balanced', top_n_validation=10, validation_weight=0.5):
    """
    Test multiple parameter combinations for a strategy to find optimal settings for nameless betting system.

    This function supports two optimization approaches:
    1. Grid Search: Tests all combinations of parameters (use_genetic=False)
    2. Genetic Algorithm: Uses evolutionary approach to find optimal parameters (use_genetic=True)

    Both approaches can be run with parallel processing (use_parallel=True) to speed up execution
    by distributing work across multiple CPU cores.

    When use_cross_validation=True, the historical data is split into training and validation sets.
    Parameters are optimized on the training set, and then the best parameters are evaluated on the
    validation set to check for overfitting.

    Args:
        strategy (BettingStrategy): Strategy to optimize
        param_ranges (dict, optional): Dictionary of parameter ranges to explore.
            If None, will try to load from strategy file.
        base_params (dict, optional): Base parameters to start from. Defaults to {}.
        historical_data_df (DataFrame, optional): Historical data for simulation.
            If None, will load from default CSV file.
        left_cubes (list, optional): Initial left cubes configuration.
            Defaults to [4,5,6,7,8,9,10,11,12,13,14].
        right_cubes (list, optional): Initial right cubes configuration.
            Defaults to [44,40,36,33,30,27,24,21,18,17,16,15].
        stop_loss (float, optional): Stop loss threshold. Defaults to -4000.
        use_parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        use_genetic (bool, optional): Whether to use genetic algorithm. Defaults to False.
        population_size (int, optional): Population size for genetic algorithm. Defaults to 20.
        generations (int, optional): Number of generations for genetic algorithm. Defaults to 5.
        mutation_rate (float, optional): Mutation rate for genetic algorithm. Defaults to 0.2.
        use_cross_validation (bool, optional): Whether to use cross-validation. Defaults to True.
        validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.3.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        fitness_function (str, optional): The fitness function to use. Options:
            - 'balanced': Balanced approach (default)
            - 'robust': Focus on consistency and risk management
            - 'profit_focused': Focus on maximizing profit
            - 'consistency': Focus on consistency of returns
            - 'anti_overfitting': Specifically designed to combat overfitting
        top_n_validation (int, optional): Number of top parameter sets to evaluate on validation data. Defaults to 10.
        validation_weight (float, optional): Weight given to validation fitness vs training fitness (0-1). Defaults to 0.5.

    Returns:
        tuple: (best_params, all_results, validation_results) - Best parameter set, full results DataFrame, and validation results (if available)

    Examples:
        # Use parallel grid search with parameter ranges from strategy file
        best_params, results, validation_results = test_parameter_combinations(
            strategy=strategy,
            use_parallel=True,
            use_genetic=False
        )

        # Use parallel genetic algorithm with parameter ranges from strategy file
        best_params, results, validation_results = test_parameter_combinations(
            strategy=strategy,
            use_parallel=True,
            use_genetic=True,
            population_size=20,
            generations=5,
            fitness_function='robust'  # Use the robust fitness function
        )

        # Use genetic algorithm with anti-overfitting fitness function and cross-validation
        best_params, results, validation_results = test_parameter_combinations(
            strategy=strategy,
            use_parallel=True,
            use_genetic=True,
            population_size=50,
            generations=10,
            fitness_function='anti_overfitting',
            use_cross_validation=True,
            validation_split=0.3,
            top_n_validation=10,  # Test top 10 parameter sets on validation data
            validation_weight=0.5  # Equal weight to training and validation fitness
        )
    """
    logger.info(f"Testing parameter combinations for {strategy.value}")

    # Set default values if not provided
    if base_params is None:
        base_params = {}

    if left_cubes is None:
        left_cubes = [4,5,6,7,8,9,10,11,12,13,14]

    if right_cubes is None:
        right_cubes = [44,40,36,33,30,27,24,21,18,17,16,15]

    # Load historical data if not provided
    if historical_data_df is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return None, pd.DataFrame()

        try:
            historical_data_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(historical_data_df)} lines from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None, pd.DataFrame()

    # Use consistent train/validation split across strategies
    from consistent_split import initialize_data_split, get_training_data, get_validation_data

    # Initialize the data split if not already done
    training_data, validation_data = initialize_data_split(
        historical_data_df=historical_data_df,
        validation_split=validation_split,
        random_seed=random_seed,
        use_cross_validation=use_cross_validation
    )

    logger.info(f"Using consistent train/validation split across all strategies")
    if validation_data is not None:
        logger.info(f"Training data: {len(training_data)} samples, Validation data: {len(validation_data)} samples")

    # Try to load parameter ranges from strategy file if not provided
    if param_ranges is None:
        try:
            # Import the function to get parameter ranges
            from strategies.parameter_ranges import get_parameter_ranges
            param_ranges = get_parameter_ranges(strategy)
            if not param_ranges:
                logger.warning(f"No parameter ranges defined for {strategy.value}. Using default ranges.")
                # Define some default parameter ranges based on the strategy type
                if strategy == BettingStrategy.NEURAL_OSCILLATOR:
                    param_ranges = {
                        'num_oscillators': {'min': 2, 'max': 12, 'steps': 11},
                        'coupling_strength': {'min': 0.01, 'max': 0.9, 'steps': 50},
                        'adaptation_rate': {'min': 0.01, 'max': 0.6, 'steps': 50},
                        'resonance_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
                        'phase_sync_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
                        'min_samples': {'min': 2, 'max': 8, 'steps': 7},
                        'banker_bias': {'min': 0.0, 'max': 0.2, 'steps': 20},
                        'phases': {'special': 'oscillator_phases'},
                    }
                else:
                    # Generic default parameters for other strategies
                    param_ranges = {
                        'window_size': {'min': 5, 'max': 20, 'steps': 4},
                        'confidence_threshold': {'min': 0.5, 'max': 0.9, 'steps': 5}
                    }
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load parameter ranges from strategy file: {e}")
            # Define some default parameter ranges based on the strategy type
            if strategy == BettingStrategy.NEURAL_OSCILLATOR:
                param_ranges = {
                    'num_oscillators': {'min': 2, 'max': 12, 'steps': 11},
                    'coupling_strength': {'min': 0.01, 'max': 0.9, 'steps': 50},
                    'adaptation_rate': {'min': 0.01, 'max': 0.6, 'steps': 50},
                    'resonance_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
                    'phase_sync_threshold': {'min': 0.1, 'max': 0.9, 'steps': 50},
                    'min_samples': {'min': 2, 'max': 8, 'steps': 7},
                    'banker_bias': {'min': 0.0, 'max': 0.2, 'steps': 20},
                    'phases': {'special': 'oscillator_phases'},
                }
            else:
                # Generic default parameters for other strategies
                param_ranges = {
                    'window_size': {'min': 5, 'max': 20, 'steps': 4},
                    'confidence_threshold': {'min': 0.5, 'max': 0.9, 'steps': 5}
                }

    # Determine which approach to use for parameter testing
    if use_genetic:
        logger.info(f"Using genetic algorithm for parameter search with population size {population_size} and {generations} generations")
        logger.info(f"Genetic algorithm will use {'parallel' if use_parallel else 'sequential'} processing")
        param_sets, results = _run_genetic_algorithm(
            strategy, param_ranges, base_params, training_data,  # Use training data for optimization
            left_cubes, right_cubes, stop_loss,
            population_size, generations, mutation_rate, use_parallel,
            fitness_function  # Pass the fitness function parameter
        )
    else:
        # Generate all parameter combinations
        param_sets = generate_parameter_combinations(base_params, param_ranges)
        logger.info(f"Generated {len(param_sets)} parameter combinations to test")

        # Store results for each parameter set
        results = []

        if use_parallel:
            # Use parallel processing to test parameter combinations
            from multiprocessing import Pool, cpu_count
            print(f"Using parallel processing with {cpu_count()} cores")

            # Create arguments for parallel processing
            args_list = [
                (strategy, param_set, training_data, left_cubes, right_cubes, stop_loss, param_ranges, i, len(param_sets))
                for i, param_set in enumerate(param_sets)
            ]

            # Run in parallel
            try:
                # Use a process pool to run parameter tests in parallel
                with Pool(processes=cpu_count()) as pool:
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
                result = _run_single_parameter_set((strategy, param_set, training_data, left_cubes, right_cubes, stop_loss, param_ranges, i, len(param_sets)))
                if result is not None:
                    results.append(result)

        # Create DataFrame from results
        if results:
            results_df = pd.DataFrame(results)
        else:
            logger.warning("No valid results generated from parameter testing")
            return None, pd.DataFrame()

    # Print top 2 parameter sets by Sortino ratio and profitable lines percentage
    if isinstance(results, pd.DataFrame):
        results_df = results
    else:
        results_df = pd.DataFrame(results)

    if not results_df.empty:
        print("\n\nTop 2 parameter sets by Sortino ratio:")
        if 'Sortino Ratio' in results_df.columns:
            top_sortino = results_df.sort_values('Sortino Ratio', ascending=False).head(2)
            for _, row in top_sortino.iterrows():
                print(f"Sortino Ratio: {row.get('Sortino Ratio', 0):.3f}, Profit: ${row.get('Total Profit', 0):.2f}")
                print(f"Win Rate: {row.get('Win Rate', 0):.2f}%, Games/Line: {row.get('Avg Games Per Line', 0):.1f}")
                losing_lines = row.get('Losing Lines', 0)
                total_lines = row.get('Total Lines', 0)
                losing_pct = row.get('Losing Lines %', 0)
                print(f"Losing Lines: {losing_lines}/{total_lines} ({losing_pct:.1f}%)")
                print(f"Max Drawdown: ${row.get('Max Drawdown', 0):.2f}, Max Closed Drawdown: ${row.get('Max Closed Drawdown', 0):.2f}, Avg Drawdown: ${row.get('Avg Drawdown', 0):.2f}")
                print(f"Parameters: {row['Parameters']}") # Display the full parameter set
                print("-" * 60)
        else:
            logger.warning("Could not find 'Sortino Ratio' in results to determine top 2.")

        print("\nTop 2 parameter sets by profitable lines percentage:")
        if 'Completion Rate %' in results_df.columns:
            top_profit_pct = results_df.sort_values('Completion Rate %', ascending=False).head(2)
            for _, row in top_profit_pct.iterrows():
                print(f"Completion Rate: {row.get('Completion Rate %', 0):.2f}%, Profit: ${row.get('Total Profit', 0):.2f}")
                print(f"Win Rate: {row.get('Win Rate', 0):.2f}%, Games/Line: {row.get('Avg Games Per Line', 0):.1f}")
                losing_lines = row.get('Losing Lines', 0)
                total_lines = row.get('Total Lines', 0)
                losing_pct = row.get('Losing Lines %', 0)
                print(f"Losing Lines: {losing_lines}/{total_lines} ({losing_pct:.1f}%)")
                print(f"Max Drawdown: ${row.get('Max Drawdown', 0):.2f}, Max Closed Drawdown: ${row.get('Max Closed Drawdown', 0):.2f}, Avg Drawdown: ${row.get('Avg Drawdown', 0):.2f}")
                print(f"Parameters: {row['Parameters']}") # Display the full parameter set
                print("-" * 60)
        else:
            logger.warning("Could not find 'Completion Rate %' in results to determine top 2.")

        # Also print top 2 by total profit
        print("\nTop 2 parameter sets by total profit:")
        if 'Total Profit' in results_df.columns:
            top_profit = results_df.sort_values('Total Profit', ascending=False).head(2)
            for _, row in top_profit.iterrows():
                print(f"Total Profit: ${row.get('Total Profit', 0):.2f}, Sortino: {row.get('Sortino Ratio', 0):.3f}")
                print(f"Win Rate: {row.get('Win Rate', 0):.2f}%, Games/Line: {row.get('Avg Games Per Line', 0):.1f}")
                losing_lines = row.get('Losing Lines', 0)
                total_lines = row.get('Total Lines', 0)
                losing_pct = row.get('Losing Lines %', 0)
                print(f"Losing Lines: {losing_lines}/{total_lines} ({losing_pct:.1f}%)")
                print(f"Max Drawdown: ${row.get('Max Drawdown', 0):.2f}, Max Closed Drawdown: ${row.get('Max Closed Drawdown', 0):.2f}, Avg Drawdown: ${row.get('Avg Drawdown', 0):.2f}")
                print(f"Parameters: {row['Parameters']}") # Display the full parameter set
                print("-" * 60)
        else:
            logger.warning("Could not find 'Total Profit' in results to determine top 2.")
    else:
        logger.warning("No valid results generated from parameter testing")

    # For genetic algorithm results, we need to handle the different format
    if use_genetic:
        # Check if we have the fitness column
        if 'Fitness' in results_df.columns:
            logger.info("Processing genetic algorithm results")

            # If cross-validation is enabled and we have enough parameter sets, test top N on validation data
            if use_cross_validation and validation_data is not None and len(results_df) >= top_n_validation:
                logger.info(f"Testing top {top_n_validation} parameter sets on validation data")

                # Get top N parameter sets by training fitness
                top_n_params = results_df.sort_values('Fitness', ascending=False).head(top_n_validation)

                # Store validation results for each parameter set
                validation_results_list = []
                combined_scores = []

                # Test each parameter set on validation data
                for i, (_, row) in enumerate(top_n_params.iterrows()):
                    param_set = row['Parameters']
                    train_fitness = row['Fitness']

                    logger.info(f"Testing parameter set {i+1}/{top_n_validation} on validation data")

                    # Run simulation with this parameter set on validation data
                    # Use the consistent validation data from our split
                    validation_data_to_use = get_validation_data()

                    summary_df, all_results = simulate_nameless_strategies(
                        validation_data_to_use,
                        selected_strategies=[strategy],
                        initial_left_cubes=left_cubes,
                        initial_right_cubes=right_cubes,
                        use_optimized_params=True,
                        override_params={strategy: [param_set]},
                        stop_loss=stop_loss
                    )

                    if summary_df is not None and not summary_df.empty:
                        # Get basic metrics from summary DataFrame
                        val_result = summary_df.iloc[0].to_dict()

                        # Calculate additional metrics using our helper function
                        additional_metrics = _calculate_additional_metrics(strategy, all_results)

                        # Merge the additional metrics with the basic metrics
                        val_result.update(additional_metrics)

                        # Calculate validation fitness
                        val_sortino = val_result.get('Sortino Ratio', 0)
                        val_profit = val_result.get('Total Profit', 0)

                        val_fitness = _calculate_fitness(
                            val_result,
                            fitness_function,
                            val_sortino,
                            val_result.get('Completion Rate %', 0),
                            val_profit,
                            val_result.get('Win Rate', 0),
                            val_result.get('Max Drawdown', 0),
                            val_result.get('Losing Lines', 0),
                            val_result.get('Total Lines', 0),
                            val_result.get('Profitable Lines %', 0),
                            val_result.get('Betting Frequency %', 0),
                            val_result.get('Avg Games Per Line', 0)
                        )

                        # Calculate combined score (weighted average of training and validation fitness)
                        combined_score = (1 - validation_weight) * train_fitness + validation_weight * val_fitness

                        # Store results
                        validation_results_list.append(val_result)
                        combined_scores.append({
                            'param_set': param_set,
                            'train_fitness': train_fitness,
                            'val_fitness': val_fitness,
                            'combined_score': combined_score,
                            'train_profit': row.get('Total Profit', 0),
                            'val_profit': val_profit,
                            'train_sortino': row.get('Sortino Ratio', 0),
                            'val_sortino': val_sortino,
                            'index': i
                        })

                        logger.info(f"Parameter set {i+1}: Train fitness: {train_fitness:.3f}, Val fitness: {val_fitness:.3f}, Combined: {combined_score:.3f}")
                    else:
                        logger.warning(f"No validation results for parameter set {i+1}")

                # Find the parameter set with the best combined score
                if combined_scores:
                    best_combined = max(combined_scores, key=lambda x: x['combined_score'])
                    best_params = best_combined['param_set']
                    best_idx = best_combined['index']

                    # Store validation results for the best parameter set
                    validation_results = validation_results_list[best_idx]

                    logger.info(f"Best parameter set by combined score (train/val weight: {1-validation_weight:.1f}/{validation_weight:.1f}):")
                    logger.info(f"Train fitness: {best_combined['train_fitness']:.3f}, Val fitness: {best_combined['val_fitness']:.3f}, Combined: {best_combined['combined_score']:.3f}")
                    logger.info(f"Train profit: ${best_combined['train_profit']:.2f}, Val profit: ${best_combined['val_profit']:.2f}")
                else:
                    # Fallback to best training fitness if validation failed
                    logger.warning("Validation failed for all parameter sets, falling back to best training fitness")
                    best_idx = results_df['Fitness'].idxmax()
                    best_params = results_df.loc[best_idx, 'Parameters']
            else:
                # If cross-validation is disabled or we don't have enough parameter sets, use best training fitness
                try:
                    best_idx = results_df['Fitness'].idxmax()
                    best_params = results_df.loc[best_idx, 'Parameters']
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

    # Validate the best parameters on the validation set if cross-validation is enabled
    # Skip this step if we already did validation with top-N parameter sets
    if use_genetic and use_cross_validation and 'validation_results_list' in locals() and validation_results_list:
        logger.info("Skipping additional validation as we already validated with top-N parameter sets")
        # validation_results should already be set to the best one from the top-N validation
    elif use_cross_validation and validation_data is not None and best_params:
        validation_results = None
        logger.info("Validating best parameters on validation set...")

        # For genetic algorithm, we have a single best parameter set
        if use_genetic and isinstance(best_params, dict) and not isinstance(best_params.get('Total Profit', None), dict):
            # Run simulation with best parameters on validation set
            summary_df, all_results = simulate_nameless_strategies(
                validation_data,
                selected_strategies=[strategy],
                initial_left_cubes=left_cubes,
                initial_right_cubes=right_cubes,
                use_optimized_params=True,
                override_params={strategy: [best_params]},
                stop_loss=stop_loss
            )

            if summary_df is not None and not summary_df.empty:
                # Get basic metrics from summary DataFrame
                validation_results = summary_df.iloc[0].to_dict()

                # Calculate additional metrics using our helper function
                additional_metrics = _calculate_additional_metrics(strategy, all_results)

                # Merge the additional metrics with the basic metrics
                validation_results.update(additional_metrics)

                # Compare training and validation performance
                train_fitness = results_df.loc[best_idx, 'Fitness']
                train_profit = results_df.loc[best_idx, 'Total Profit']
                train_sortino = results_df.loc[best_idx, 'Sortino Ratio']

                val_profit = validation_results.get('Total Profit', 0)
                val_sortino = validation_results.get('Sortino Ratio', 0)

                # Calculate validation fitness using the same fitness function as training
                val_fitness = _calculate_fitness(
                    validation_results,
                    fitness_function,
                    val_sortino,
                    validation_results.get('Completion Rate %', 0),
                    val_profit,
                    validation_results.get('Win Rate', 0),
                    validation_results.get('Max Drawdown', 0),
                    validation_results.get('Losing Lines', 0),
                    validation_results.get('Total Lines', 0),
                    validation_results.get('Profitable Lines %', 0),
                    validation_results.get('Betting Frequency %', 0),
                    validation_results.get('Avg Games Per Line', 0)
                )

                logger.info(f"\nValidation Results for Best Parameters:")
                logger.info(f"Training Fitness: {train_fitness:.3f}, Validation Fitness: {val_fitness:.3f}")
                logger.info(f"Training Profit: ${train_profit:.2f}, Validation Profit: ${val_profit:.2f}")
                logger.info(f"Training Sortino: {train_sortino:.3f}, Validation Sortino: {val_sortino:.3f}")

                # Check for overfitting
                if val_fitness < train_fitness * 0.7:  # If validation fitness is less than 70% of training fitness
                    logger.warning("\nPOTENTIAL OVERFITTING DETECTED")
                    logger.warning(f"The model performs significantly worse on validation data (fitness: {val_fitness:.3f}) than on training data (fitness: {train_fitness:.3f})")
                    logger.warning("Consider using more conservative parameters or collecting more diverse training data.")
                elif val_profit < 0 and train_profit > 0:
                    logger.warning("\nPOTENTIAL OVERFITTING DETECTED")
                    logger.warning(f"The model is profitable on training data (${train_profit:.2f}) but loses money on validation data (${val_profit:.2f})")
                    logger.warning("Consider using more conservative parameters or collecting more diverse training data.")
                else:
                    logger.info("\nModel generalizes well to validation data.")

        # For grid search, we have multiple best parameter sets (one for each metric)
        elif not use_genetic and isinstance(best_params, dict):
            for metric, params in best_params.items():
                if params is None:
                    continue

                logger.info(f"\nValidating best parameters for {metric} on validation set...")

                # Run simulation with best parameters on validation set
                summary_df, all_results = simulate_nameless_strategies(
                    validation_data,
                    selected_strategies=[strategy],
                    initial_left_cubes=left_cubes,
                    initial_right_cubes=right_cubes,
                    use_optimized_params=True,
                    override_params={strategy: [params]},
                    stop_loss=stop_loss
                )

                if summary_df is not None and not summary_df.empty:
                    # Get basic metrics from summary DataFrame
                    val_result = summary_df.iloc[0].to_dict()

                    # Calculate additional metrics using our helper function
                    additional_metrics = _calculate_additional_metrics(strategy, all_results)

                    # Merge the additional metrics with the basic metrics
                    val_result.update(additional_metrics)

                    # Find the training row with these parameters
                    train_row = None
                    for _, row in results_df.iterrows():
                        if row['Parameters'] == params:
                            train_row = row
                            break

                    if train_row is not None:
                        # Compare training and validation performance
                        train_value = train_row.get(metric, 0)
                        val_value = val_result.get(metric, 0)

                        logger.info(f"Best by {metric}:")
                        logger.info(f"Training {metric}: {train_value:.3f}, Validation {metric}: {val_value:.3f}")
                        logger.info(f"Training Profit: ${train_row.get('Total Profit', 0):.2f}, Validation Profit: ${val_result.get('Total Profit', 0):.2f}")

                        # Check for overfitting
                        if val_value < train_value * 0.7 and metric != 'Total Profit':  # If validation metric is less than 70% of training metric
                            logger.warning("POTENTIAL OVERFITTING DETECTED")
                            logger.warning(f"The model performs significantly worse on validation data ({metric}: {val_value:.3f}) than on training data ({metric}: {train_value:.3f})")
                        elif metric == 'Total Profit' and val_value < 0 and train_value > 0:
                            logger.warning("POTENTIAL OVERFITTING DETECTED")
                            logger.warning(f"The model is profitable on training data (${train_value:.2f}) but loses money on validation data (${val_value:.2f})")

    # Return best parameters, results dataframe, and validation results (if available)
    return best_params, results_df, validation_results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nameless_simulator.log'),
            logging.StreamHandler()
        ]
    )

    # Import necessary modules
    import pandas as pd
    from strategies.betting_strategy import BettingStrategy

    # Load historical data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines_combined.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        exit(1)

    try:
        historical_data_df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(historical_data_df)} lines from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        exit(1)

    historical_data_df = historical_data_df[historical_data_df['all_outcomes_first_shoe'].apply(lambda x: isinstance(x, str) and len(x) >= 45)]
    logger.info(f"Filtered to {len(historical_data_df)} lines with 45+ outcomes")

    # Define strategies to test
    test_strategies = [
        #BettingStrategy.ORIGINAL,
        #BettingStrategy.FREQUENCY_ANALYSIS,
        #BettingStrategy.HYBRID_PATTERN,
        #BettingStrategy.HYBRID_FREQUENCY_VOLATILITY,
        #BettingStrategy.VOLATILITY_ADAPTIVE,
        #BettingStrategy.PATTERN_BASED,
        #BettingStrategy.PATTERN_INTERRUPTION,
        #BettingStrategy.DEEP_Q_NETWORK,
        #BettingStrategy.THOMPSON_SAMPLING,
        #BettingStrategy.SEQUENTIAL_PATTERN_MINING,
        #BettingStrategy.TIME_SERIES_FORECASTING,
        #BettingStrategy.BAYESIAN_INFERENCE,
        #BettingStrategy.ADAPTIVE_BIAS,
        #BettingStrategy.ADAPTIVE_MOMENTUM,
        #BettingStrategy.BAYESIAN_NETWORK,
        #BettingStrategy.CHAOS_THEORY,
        #BettingStrategy.FRACTAL_ANALYSIS,
        #BettingStrategy.INFORMATION_THEORY,
        #BettingStrategy.MAJORITY_LAST_N,
        #BettingStrategy.MOMENTUM_OSCILLATOR,
        BettingStrategy.NEURAL_OSCILLATOR
        #BettingStrategy.QUANTUM_INSPIRED,
        #BettingStrategy.REINFORCEMENT_META_LEARNING,
        #BettingStrategy.SYMBOLIC_DYNAMICS,
        #BettingStrategy.TREND_CONFIRMATION,
        #BettingStrategy.TRANSFER_LEARNING,
        #BettingStrategy.REINFORCEMENT_LEARNING,
        #BettingStrategy.SEQUENTIAL_PATTERN_MINING,
        #BettingStrategy.ADVANCED_CHAOS_THEORY,
        #BettingStrategy.ML_STRATEGY,
        #BettingStrategy.MULTI_SCALE_MOMENTUM,
        #BettingStrategy.DRIFT_DETECTION,
        #BettingStrategy.CONDITIONAL_PROBABILITY_CHAIN,
        #BettingStrategy.STATISTICAL_ARBITRAGE,
        #BettingStrategy.ADAPTIVE_PATTERN_RECOGNITION
    ]

    # Define initial cube configurations to test
    cube_configs = [
        ([4,5,6,7,8,9,10,11,12,13,14], [44,40,36,33,30,27,24,21,18,17,16,15])
    ]
    cube_configs = [
        ([2,3,4,5,6,7,8], [24,21,18,17,16,15,14,13])
    ]
    # Example of running parameter optimization with genetic algorithm
    # Uncomment to run optimization

    # Dictionary of fitness functions to try
    fitness_functions = {
        'alligator_fitness': 'Focus on both profit and consistency',
        'balanced': 'Balanced approach (default)',
        'robust': 'Focus on consistency and risk management',
        'profit_focused': 'Focus on maximizing profit',
        'consistency': 'Focus on consistency of returns',
        #'anti_overfitting': 'Specifically designed to combat overfitting'
    }
    # Try each fitness function
    '''
    for fitness_name, fitness_desc in fitness_functions.items():
        print(f"\n\n{'='*80}")
        print(f"Testing {fitness_name} fitness function: {fitness_desc}")
        print(f"{'='*80}\n")

        for strategy in test_strategies:
            print(f"\nOptimizing parameters for {strategy.value} strategy with {fitness_name} fitness function")
            # Run parameter optimization with cross-validation
            best_params, results, val_results = test_parameter_combinations(
                strategy=strategy,
                historical_data_df=historical_data_df,
                left_cubes=cube_configs[0][0],
                right_cubes=cube_configs[0][1],
                use_parallel=True,
                use_genetic=True,
                population_size=500,
                generations=10,
                use_cross_validation=True,  # Enable cross-validation to detect overfitting
                validation_split=0.3,       # Use 30% of data for validation
                random_seed=42,             # Set random seed for reproducibility
                fitness_function=fitness_name,
                top_n_validation=20,  # Test top 10 parameter sets on validation data
                validation_weight=0.75,  # Equal weight to training and validation fitness
                stop_loss=-1000
            )

            # Validation results are now directly available from the function return
            validation_results = val_results

            # Save the optimized parameters to a file for this strategy
            if best_params:
                output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                os.makedirs(output_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                params_filename = f"optimized_params_{strategy.value}_{fitness_name}_{timestamp}.txt"
                params_path = os.path.join(output_dir, params_filename)

                # Check if we have validation results and if both training and validation are profitable
                train_profit = 0
                val_profit = 0

                # Find the row with the best fitness
                if isinstance(results, pd.DataFrame) and 'Fitness' in results.columns:
                    best_idx = results['Fitness'].idxmax()
                    best_row = results.loc[best_idx]
                    train_profit = best_row.get('Total Profit', 0)

                # Get validation profit if available
                if 'validation_results' in locals() and validation_results is not None:
                    val_profit = validation_results.get('Total Profit', 0)

                # Only save if both training and validation are profitable
                if train_profit <= 0 or (validation_results is not None and val_profit <= 0):
                    print(f"\nSkipping save for {strategy.value} with {fitness_name} fitness - not profitable on both datasets")
                    print(f"Training profit: ${train_profit:.2f}, Validation profit: ${val_profit:.2f}")
                    continue

                with open(params_path, 'w') as f:
                    f.write(f"Optimized parameters for {strategy.value} strategy using {fitness_name} fitness function:\n")
                    f.write(f"{best_params}\n\n")
                    f.write("Top metrics:\n")

                    # Write detailed metrics
                    f.write(f"Fitness: {best_row.get('Fitness')}\n")
                    f.write(f"Sortino Ratio: {best_row.get('Sortino Ratio', 0):.3f}\n")
                    f.write(f"Total Profit: ${train_profit:.2f}\n")
                    f.write(f"Win Rate: {best_row.get('Win Rate', 0):.2f}%\n")
                    f.write(f"Completion Rate: {best_row.get('Completion Rate %', 0):.2f}%\n")
                    f.write(f"Avg Games Per Line: {best_row.get('Avg Games Per Line', 0):.1f}\n")

                    # Add losing lines information
                    losing_lines = best_row.get('Losing Lines', 0)
                    total_lines = best_row.get('Total Lines', 0)
                    losing_pct = best_row.get('Losing Lines %', 0)
                    f.write(f"Losing Lines: {losing_lines}/{total_lines} ({losing_pct:.1f}%)\n")

                    f.write(f"Max Drawdown: ${best_row.get('Max Drawdown', 0):.2f}\n")
                    f.write(f"Max Closed Drawdown: ${best_row.get('Max Closed Drawdown', 0):.2f}\n")
                    f.write(f"Avg Drawdown: ${best_row.get('Avg Drawdown', 0):.2f}\n")
                    f.write(f"Profitable Lines: {best_row.get('Profitable Lines %', 0):.2f}%\n")
                    f.write(f"Betting Frequency: {best_row.get('Betting Frequency %', 0):.2f}%\n")

                    # Add validation results and overfitting analysis if available
                    if 'validation_results' in locals() and validation_results is not None:
                        f.write("\n" + "="*50 + "\n")
                        f.write("VALIDATION RESULTS\n")
                        f.write("="*50 + "\n")

                        # Training vs Validation metrics
                        train_fitness = best_row.get('Fitness')
                        train_profit = best_row.get('Total Profit', 0)
                        train_sortino = best_row.get('Sortino Ratio', 0)

                        val_profit = validation_results.get('Total Profit', 0)
                        val_sortino = validation_results.get('Sortino Ratio', 0)

                        # Calculate validation fitness using the same formula
                        val_fitness = _calculate_fitness(
                            validation_results,
                            fitness_name,  # Use the current fitness function name
                            val_sortino,
                            validation_results.get('Completion Rate %', 0),
                            val_profit,
                            validation_results.get('Win Rate', 0),
                            validation_results.get('Max Drawdown', 0),
                            validation_results.get('Losing Lines', 0),
                            validation_results.get('Total Lines', 0),
                            validation_results.get('Profitable Lines %', 0),
                            validation_results.get('Betting Frequency %', 0),
                            validation_results.get('Avg Games Per Line', 0)
                        )

                        # Write comparison
                        f.write(f"Training Fitness: {train_fitness:.3f}, Validation Fitness: {val_fitness:.3f}\n")
                        f.write(f"Training Profit: ${train_profit:.2f}, Validation Profit: ${val_profit:.2f}\n")
                        f.write(f"Training Sortino: {train_sortino:.3f}, Validation Sortino: {val_sortino:.3f}\n")

                        # Write validation metrics
                        f.write("\nDetailed Validation Metrics:\n")
                        f.write(f"Win Rate: {validation_results.get('Win Rate', 0):.2f}%\n")
                        f.write(f"Completion Rate: {validation_results.get('Completion Rate %', 0):.2f}%\n")
                        f.write(f"Avg Games Per Line: {validation_results.get('Avg Games Per Line', 0):.1f}\n")

                        # Add losing lines information for validation
                        val_losing_lines = validation_results.get('Losing Lines', 0)
                        val_total_lines = validation_results.get('Total Lines', 0)
                        val_losing_pct = validation_results.get('Losing Lines %', 0)
                        f.write(f"Losing Lines: {val_losing_lines}/{val_total_lines} ({val_losing_pct:.1f}%)\n")

                        f.write(f"Max Drawdown: ${validation_results.get('Max Drawdown', 0):.2f}\n")
                        f.write(f"Max Closed Drawdown: ${validation_results.get('Max Closed Drawdown', 0):.2f}\n")
                        f.write(f"Profitable Lines: {validation_results.get('Profitable Lines %', 0):.2f}%\n")

                        # Check for overfitting and add warning if detected
                        f.write("\n" + "="*50 + "\n")
                        f.write("OVERFITTING ANALYSIS\n")
                        f.write("="*50 + "\n")

                        if val_fitness < train_fitness * 0.7:  # If validation fitness is less than 70% of training fitness
                            f.write("POTENTIAL OVERFITTING DETECTED\n")
                            f.write(f"The model performs significantly worse on validation data (fitness: {val_fitness:.3f}) than on training data (fitness: {train_fitness:.3f})\n")
                            f.write("Consider using more conservative parameters or collecting more diverse training data.\n")
                        elif val_profit < 0 and train_profit > 0:
                            f.write("POTENTIAL OVERFITTING DETECTED\n")
                            f.write(f"The model is profitable on training data (${train_profit:.2f}) but loses money on validation data (${val_profit:.2f})\n")
                            f.write("Consider using more conservative parameters or collecting more diverse training data.\n")
                        else:
                            f.write("Model generalizes well to validation data.\n")

                        # Add fitness function information
                        f.write("\nFitness Function: " + fitness_name + "\n")
                        fitness_descriptions = {
                            'balanced': 'Balanced approach that considers profit, Sortino ratio, win rate, and completion rate',
                            'robust': 'Focus on consistency and risk management, heavily penalizes drawdowns and losing lines',
                            'profit_focused': 'Aggressive approach that prioritizes total profit (may lead to more overfitting)',
                            'consistency': 'Prioritizes strategies that win consistently with lower variance',
                            'anti_overfitting': 'Specifically designed to combat overfitting by penalizing extreme strategies'
                        }
                        f.write(fitness_descriptions.get(fitness_name, "Custom fitness function") + "\n")

                print(f"Saved optimized parameters to {params_path}")

            print(f"Best parameters for {strategy.value} with {fitness_name} fitness: {best_params}")
            print(f"{'='*60}\n")
    
    # Exit after running all fitness functions and strategies
    exit(0)
    '''
    # Run simulations for each cube configuration
    for left_cubes, right_cubes in cube_configs:
        cube_desc = f"L:{left_cubes}, R:{right_cubes}"
        logger.info(f"\nRunning nameless simulation with cube configuration: {cube_desc}")

        try:
            summary_df, results = simulate_nameless_strategies(
                historical_data_df,
                selected_strategies=test_strategies,
                initial_left_cubes=left_cubes,
                initial_right_cubes=right_cubes,
                use_optimized_params=False,
                stop_loss=-1000  # Stop the line if PnL goes below -4000
            )

            if len(summary_df) == 0:
                logger.warning("No results were generated. Check the log for details.")
            else:
                logger.info(f"Simulation completed successfully with {len(summary_df)} strategy variants.")

                # Print top 5 strategies by profit
                top_5 = summary_df.head(5)
                logger.info("\nTop 5 strategies by profit:")
                for _, row in top_5.iterrows():
                    logger.info(f"{row['Strategy']} ({row['Description']}): ${row['Total Profit']:.2f}, Win Rate: {row['Win Rate']:.2f}%, Sortino: {row['Sortino Ratio']:.2f}")

                # Save results to CSV
                output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                os.makedirs(output_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"nameless_results_{timestamp}_{cube_desc.replace(':', '_').replace(', ', '_')}.csv"
                csv_path = os.path.join(output_dir, csv_filename)

                # Convert to CSV-friendly format
                csv_df = summary_df.drop(columns=['All Results', 'Parameters'])

                # Add additional columns for better analysis
                csv_df['Profit Per Completed Line'] = csv_df.apply(
                    lambda row: row['Total Profit'] / row['Lines Completed'] if row['Lines Completed'] > 0 else 0,
                    axis=1
                )

                # Format percentage columns
                for col in ['Win Rate', 'Betting Frequency %', 'Completion Rate %']:
                    if col in csv_df.columns:
                        csv_df[col] = csv_df[col].round(2)

                # Format monetary columns
                for col in ['Total Profit', 'Commission Paid', 'Profit Per Line', 'Profit Per Bet', 'Profit Per Completed Line']:
                    if col in csv_df.columns:
                        csv_df[col] = csv_df[col].round(2)

                # Format ratio columns
                for col in ['Sharpe Ratio', 'Sortino Ratio']:
                    if col in csv_df.columns:
                        csv_df[col] = csv_df[col].round(3)

                csv_df.to_csv(csv_path, index=False)
                logger.info(f"Results saved to {csv_path}")

                # Print detailed statistics for the best strategy
                if not csv_df.empty:
                    best_strategy = csv_df.iloc[0]
                    logger.info(f"\nBest strategy: {best_strategy['Strategy']} ({best_strategy['Description']})")
                    logger.info(f"Total Profit: ${best_strategy['Total Profit']:.2f}")
                    logger.info(f"Win Rate: {best_strategy['Win Rate']:.2f}%")
                    logger.info(f"Sortino Ratio: {best_strategy['Sortino Ratio']:.3f}")
                    logger.info(f"Lines Completed: {best_strategy['Lines Completed']}/{best_strategy['Total Lines']} ({best_strategy['Completion Rate %']:.2f}%)")
                    logger.info(f"Profit Per Completed Line: ${best_strategy['Profit Per Completed Line']:.2f}")
                    logger.info(f"Profit Per Bet: ${best_strategy['Profit Per Bet']:.2f}")
                    logger.info(f"Max Consecutive Wins/Losses: {best_strategy['Max Consecutive Wins']}/{best_strategy['Max Consecutive Losses']}")
                    logger.info(f"Betting Frequency: {best_strategy['Betting Frequency %']:.2f}%")

        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            logger.debug(traceback.format_exc())

    # Print information about available fitness functions
    print("\n" + "="*80)
    print("AVAILABLE FITNESS FUNCTIONS FOR OPTIMIZATION")
    print("="*80)
    print("\nThe system supports multiple fitness functions for parameter optimization:")

    fitness_functions = {
        'balanced': 'Balanced approach that considers profit, Sortino ratio, win rate, and completion rate',
        'robust': 'Focus on consistency and risk management, heavily penalizes drawdowns and losing lines',
        'profit_focused': 'Aggressive approach that prioritizes total profit (may lead to more overfitting)',
        'consistency': 'Prioritizes strategies that win consistently with lower variance',
        'anti_overfitting': 'Specifically designed to combat overfitting by penalizing extreme strategies'
    }

    for name, desc in fitness_functions.items():
        print(f"\n- {name}: {desc}")

    print("\nTo use a specific fitness function, add the fitness_function parameter to test_parameter_combinations:")
    print("\nbest_params, results, validation_results = test_parameter_combinations(")
    print("    strategy=strategy,")
    print("    use_genetic=True,")
    print("    fitness_function='robust',  # Choose the fitness function here")
    print("    use_cross_validation=True,  # Enable cross-validation to detect overfitting")
    print("    top_n_validation=10,        # Test top 10 parameter sets on validation data")
    print("    validation_weight=0.5       # Equal weight to training and validation fitness")
    print(")")
