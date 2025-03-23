#!/usr/bin/env python3
"""
Baccarat Line Simulator

This program simulates betting on historical baccarat lines using the nameless client.
It reads data from finished_lines.csv and plays through each line with a specified strategy.
"""

import os
import sys
import time
import logging
import pandas as pd
import argparse
from datetime import datetime
import csv
import random

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required modules from the main project
from core.nameless import (
    press_win_btn, press_tie_btn, press_loss_btn,
    press_player_start_btn, press_banker_start_btn, press_player_only_btn, press_banker_only_btn,
    press_new_line_btn, press_end_line_btn, press_reduce_btn, is_line_done
)
from core.screencapture import capture_nameless_betbox, capture_nameless_cubes
from core.ocr import extract_bet_size, extract_cubes_and_numbers
from core.strategy import (
    play_mode, check_for_end_line, determine_chip_amount, 
    get_outcomes_without_ties, get_streaks, bad_streaks_threshold_hit,
    cube_threshold_hit, get_last_6_without_ties
)
from main_sim import GameSimulator, BettingStrategy, Bet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, 'line_simulator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimulationContext:
    """Context class to simulate the state machine context used in the regular bot"""
    
    def __init__(self, initial_mode, outcomes=None):
        self.game = SimulationGame(initial_mode, outcomes)
        self.table = SimulationTable()
        
    def create_bet(self, side, size):
        """Create a new bet object"""
        return Bet(side, size)
        
    def get_total_pnl(self):
        """Get the total profit/loss"""
        return self.game.total_pnl
        
    def export_line_to_csv(self):
        """Mock method for exporting line to CSV"""
        pass
        
    def second_shoe_mode(self):
        """Mock method for switching to second shoe mode"""
        self.game.is_second_shoe = True
        
class SimulationGame:
    """Simulation game state to mimic the regular bot's game state"""
    
    def __init__(self, initial_mode, outcomes=None):
        self.initial_mode = initial_mode
        self.current_mode = initial_mode
        self.outcomes = outcomes or []
        self.is_second_shoe = False
        self.current_bet = None
        self.last_bet = None
        self.total_pnl = 0
        self.end_line_reason = None
        self.game_result = None
        
class SimulationTable:
    """Simulation table to mimic the regular bot's table state"""
    
    def __init__(self):
        self.line_start_time = datetime.now()
        self.bet_manager = SimulationBetManager()
        
class SimulationBetManager:
    """Simulation bet manager to mimic the regular bot's bet manager"""
    
    def __init__(self):
        self.bets = []
        self.ties = 0
        
    def add_bet(self, bet):
        """Add a bet to the manager"""
        self.bets.append(bet)
        if bet.result == 'T':
            self.ties += 1
            
    def get_all_bets(self):
        """Get all bets"""
        return self.bets
        
    def get_number_of_ties(self):
        """Get the number of ties"""
        return self.ties

class LineSimulator:
    """Main simulator class for replaying baccarat lines using the nameless client"""
    
    def __init__(self, csv_path, strategy=BettingStrategy.ORIGINAL, strategy_params=None, 
                 start_index=0, end_index=None, delay=1.0):
        """
        Initialize the line simulator
        
        Args:
            csv_path (str): Path to the CSV file with finished lines
            strategy (BettingStrategy): Strategy to use for betting
            strategy_params (dict): Parameters for the strategy
            start_index (int): Index of the first line to simulate
            end_index (int): Index of the last line to simulate (None for all)
            delay (float): Delay between actions in seconds
        """
        self.csv_path = csv_path
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.start_index = start_index
        self.end_index = end_index
        self.delay = delay
        self.results = []
        
        # Load the CSV data
        self.lines_df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.lines_df)} lines from {csv_path}")
        
        # Calculate the actual end_index if not provided
        if self.end_index is None:
            self.end_index = len(self.lines_df) - 1
        
        # Validate indices
        if self.start_index < 0 or self.start_index >= len(self.lines_df):
            raise ValueError(f"Invalid start_index: {self.start_index}. Must be between 0 and {len(self.lines_df)-1}.")
        if self.end_index < self.start_index or self.end_index >= len(self.lines_df):
            raise ValueError(f"Invalid end_index: {self.end_index}. Must be between {self.start_index} and {len(self.lines_df)-1}.")
            
        logger.info(f"Will simulate lines from index {self.start_index} to {self.end_index}")
        logger.info(f"Using strategy: {strategy.value} with parameters: {strategy_params}")

    def simulate_all_lines(self):
        """Simulate all selected lines from the CSV file"""
        overall_results = {
            'total_profit': 0,
            'lines_played': 0,
            'winning_lines': 0,
            'losing_lines': 0,
            'start_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        results_filename = f"results\simulation_results_{overall_results['start_time']}.csv"
        results_path = os.path.join(current_dir, results_filename)
        
        # Create results CSV header
        with open(results_path, 'w', newline='') as csvfile:
            fieldnames = ['line_index', 'timestamp', 'initial_mode', 'outcomes', 
                         'bets_placed', 'wins', 'losses', 'ties', 'profit', 'duration', 'end_reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        try:
            # Loop through selected lines
            for i in range(self.start_index, self.end_index + 1):
                logger.info(f"===== Simulating line {i} of {self.end_index} =====")
                
                # Simulate single line
                line_result = self.simulate_line(i)
                
                # Update overall statistics
                overall_results['lines_played'] += 1
                overall_results['total_profit'] += line_result['profit']
                
                if line_result['profit'] > 0:
                    overall_results['winning_lines'] += 1
                elif line_result['profit'] < 0:
                    overall_results['losing_lines'] += 1
                
                # Append to CSV
                with open(results_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(line_result)
                
                # Pause between lines
                time.sleep(self.delay * 2)
                logger.info(f"Line {i} completed")
                logger.info(f"Total profit so far: {overall_results['total_profit']}")
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            # Print final statistics
            logger.info("===== Simulation Completed =====")
            logger.info(f"Lines played: {overall_results['lines_played']}")
            logger.info(f"Winning lines: {overall_results['winning_lines']}")
            logger.info(f"Losing lines: {overall_results['losing_lines']}")
            logger.info(f"Total profit: {overall_results['total_profit']}")
            
            if overall_results['lines_played'] > 0:
                win_rate = overall_results['winning_lines'] / overall_results['lines_played'] * 100
                logger.info(f"Win rate: {win_rate:.2f}%")
            
            logger.info(f"Results saved to: {results_path}")
            
        return overall_results
            
    def simulate_line(self, line_index):
        """
        Simulate a single line from the CSV
        
        Args:
            line_index (int): Index of the line to simulate
            
        Returns:
            dict: Results of the simulation
        """
        line_data = self.lines_df.iloc[line_index]
        
        # Extract data from the line
        timestamp = line_data['timestamp']
        initial_mode = line_data['initial_mode'] 
        first_6 = list(line_data['first_6_outcomes'])
        
        # Get all outcomes, prioritizing first shoe outcomes
        if pd.notna(line_data['all_outcomes_first_shoe']) and len(line_data['all_outcomes_first_shoe']) > 0:
            all_outcomes = list(line_data['all_outcomes_first_shoe'])
        elif pd.notna(line_data['all_outcomes_second_shoe']) and len(line_data['all_outcomes_second_shoe']) > 0:
            all_outcomes = list(line_data['all_outcomes_second_shoe'])
        else:
            logger.warning(f"No outcomes found for line {line_index}, skipping")
            return {
                'line_index': line_index,
                'timestamp': timestamp,
                'initial_mode': initial_mode,
                'outcomes': "",
                'bets_placed': 0,
                'wins': 0,
                'losses': 0,
                'ties': 0,
                'profit': 0,
                'duration': 0,
                'end_reason': "no outcomes found"
            }
        
        logger.info(f"Line {line_index} - Initial mode: {initial_mode}, Outcomes: {all_outcomes}")
        
        # Start a new line in the nameless client
        logger.info(f"Starting new line")
        time.sleep(self.delay)
        
        # Set initial mode
        if initial_mode == "PPP":
            press_player_start_btn()
        else:  # BBB
            press_banker_start_btn()
        time.sleep(self.delay*3)
        
        # Create a game simulator with the first 6 outcomes and a simulation context
        simulator = GameSimulator(
            outcomes=first_6,
            initial_mode=initial_mode,
            four_start=False,  # We don't handle four_start specially in simulation
            strategy=self.strategy,
            strategy_params=self.strategy_params
        )
        
        context = SimulationContext(initial_mode, first_6)
        context.game.total_pnl = 0  # Reset PnL
        
        # Track results for this line
        results = {
            'line_index': line_index,
            'timestamp': timestamp,
            'initial_mode': initial_mode,
            'outcomes': ''.join(all_outcomes),
            'bets_placed': 0,
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'profit': 0,
            'end_reason': ""
        }
        
        try:
            # Process each outcome after the first 6
            should_end = False
            for i, outcome in enumerate(all_outcomes[6:]):
                logger.debug(f"Processing outcome {i+7}: {outcome}")
                
                # Read bet size using the same OCR function as in the regular bot
                bet_size = extract_bet_size(capture_nameless_betbox())
                logger.info(f"Current bet size: {bet_size}")
                
                # Get next bet recommendation using play_mode like in the regular bot
                next_bet_side, new_mode = simulator.simulate_single(new_outcome=outcome)
                
                # Update the mode if changed
                context.game.current_mode = new_mode
                
                # Create bet object and record it
                if next_bet_side != "SKIP":
                    current_bet = Bet(next_bet_side, bet_size)
                    context.game.current_bet = current_bet
                    context.game.last_bet = current_bet
                    simulator.last_bet = current_bet
                    results['bets_placed'] += 1
                    
                    logger.info(f"Placing bet on {next_bet_side} with size {bet_size}")
                    
                    # Determine which chips to use
                    chips = determine_chip_amount(bet_size)
                    logger.debug(f"Using chips: {chips}")
                    
                    # Add the outcome to processed outcomes
                    context.game.outcomes.append(outcome)
                    #simulator.outcomes.append(outcome)
                    
                    # Process outcome
                    if outcome == current_bet.side:
                        # Win
                        logger.info("Result: Win")
                        press_win_btn()
                        current_bet.set_result('W')
                        results['wins'] += 1
                        results['profit'] += current_bet.profit
                        context.game.total_pnl += current_bet.profit
                        simulator.total_pnl += current_bet.profit
                    elif outcome == 'T':
                        # Tie
                        logger.info("Result: Tie")
                        press_tie_btn()
                        current_bet.set_result('T')
                        results['ties'] += 1
                    else:
                        # Loss
                        logger.info("Result: Loss")
                        press_loss_btn()
                        current_bet.set_result('L')
                        results['losses'] += 1
                        results['profit'] += current_bet.profit
                        context.game.total_pnl += current_bet.profit
                        simulator.total_pnl += current_bet.profit
                    
                    # Add the bet to the bet manager
                    context.table.bet_manager.add_bet(current_bet)
                    simulator.bets.append(current_bet)
                else:
                    # Still add the outcome even if we're not betting
                    context.game.outcomes.append(outcome)
                    #simulator.outcomes.append(outcome)
                    logger.info(f"Skipping bet for outcome {outcome}")

                logger.info(f"Current PnL: {context.game.total_pnl}")
                
                time.sleep(self.delay*2)
                # Check if we should end the line using the same conditions as the regular bot
                should_end = check_for_end_line(context, use_mini_line_exit=False, use_moderate_exit=False)
                if should_end:
                    logger.info(f"Ending line due to end conditions: {context.game.end_line_reason}")
                    press_end_line_btn()
                    results['end_reason'] = context.game.end_line_reason
                    time.sleep(self.delay)
                    break
                
                # Short delay between outcomes
                time.sleep(self.delay / 2)
                
            # If we processed all outcomes without ending, end the line
            if not should_end:
                logger.info("Processed all outcomes, ending line")
                press_end_line_btn()
                results['end_reason'] = "all outcomes processed"
                
            # Wait for line to finish
            waiting_time = 0
            max_wait = 5  # Maximum wait time in seconds
            while not is_line_done() and waiting_time < max_wait:
                time.sleep(0.5)
                waiting_time += 0.5
            
            press_new_line_btn()

            if line_data['all_outcomes_first_shoe'][:6].count('T') == 0:
                original_no_games = len(line_data['all_outcomes_first_shoe'][6:])
                simulated_no_games = len(simulator.outcomes[6:])
            else:
                original_no_games = len(line_data['all_outcomes_first_shoe'][7:])
                simulated_no_games = len(simulator.outcomes[7:])

            logger.info(f"Line {line_index} completed")
            logger.info(f"Results: {results['wins']} wins, {results['losses']} losses, {results['ties']} ties")
            logger.info(f"Original line games vs. simulator: {original_no_games} original games vs. {simulated_no_games} simulated games")
            logger.info(f"Original profit vs. simulator: {line_data['profit']} original profit vs. {results['profit']} simulated profit")
            logger.info(f"End reason: {results['end_reason']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error simulating line {line_index}: {str(e)}", exc_info=True)
            # Try to end the line on error
            try:
                press_end_line_btn()
                results['end_reason'] = f"error: {str(e)}"
            except:
                pass
            return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Simulate baccarat betting on historical lines using the nameless client')
    
    parser.add_argument('--strategy', 
                        choices=[s.value for s in BettingStrategy], 
                        default='original',
                        help='The betting strategy to use')
                        
    parser.add_argument('--csv-path', 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           'results', 'finished_lines_combined.csv'),
                        help='Path to the CSV file with finished lines')
                        
    parser.add_argument('--start-index',
                        type=int, 
                        default=0,
                        help='Index of the first line to simulate')
                        
    parser.add_argument('--end-index',
                        type=int, 
                        default=None,
                        help='Index of the last line to simulate (None for all)')
                        
    parser.add_argument('--delay',
                        type=float,
                        default=0.4,
                        help='Delay between actions in seconds')
                        
    # Add strategy-specific parameters
    parser.add_argument('--streak-length', 
                       type=int,
                       help='Length of streak to consider (for streak-based strategies)')
                       
    parser.add_argument('--n', 
                       type=int,
                       help='Number of past outcomes to consider (for majority strategy)')
                       
    parser.add_argument('--pattern-length',
                       type=int,
                       help='Length of pattern to analyze (for pattern-based strategy)')
                       
    parser.add_argument('--window-size',
                       type=int,
                       help='Window size for adaptive bias strategy')
                       
    parser.add_argument('--weight-recent',
                       type=float,
                       help='Weight factor for recent outcomes in adaptive bias strategy')
    
    return parser.parse_args()

def main():
    """Main entry point for the line simulator"""
    args = parse_arguments()
    
    # Convert strategy string to enum
    strategy = BettingStrategy(args.strategy)
    
    # Build strategy parameters based on the strategy type
    strategy_params = {}
    if strategy in [BettingStrategy.FOLLOW_STREAK, BettingStrategy.COUNTER_STREAK]:
        if args.streak_length:
            strategy_params['streak_length'] = args.streak_length
    elif strategy == BettingStrategy.MAJORITY_LAST_N:
        if args.n:
            strategy_params['n'] = args.n
    elif strategy == BettingStrategy.PATTERN_BASED:
        if args.pattern_length:
            strategy_params['pattern_length'] = args.pattern_length
    elif strategy == BettingStrategy.ADAPTIVE_BIAS:
        if args.window_size:
            strategy_params['window_size'] = args.window_size
        if args.weight_recent:
            strategy_params['weight_recent'] = args.weight_recent
    
    # Create and run simulator
    simulator = LineSimulator(
        csv_path=args.csv_path,
        strategy=strategy,
        strategy_params=strategy_params,
        start_index=args.start_index,
        end_index=args.end_index,
        delay=args.delay
    )
    
    simulator.simulate_all_lines()

if __name__ == "__main__":
    main()