from Evolution.Simulator.main_sim import GameSimulator, BettingStrategy, Bet
import pandas as pd

import time
import logging
import argparse
import os

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from core.nameless import (
    press_win_btn, press_tie_btn, press_loss_btn,
    press_player_start_btn, press_banker_start_btn
)

def run_simulation(strategy=BettingStrategy.ORIGINAL, strategy_params=None, line_index=0):
    """
    Run a simulation using the selected strategy and the Nameless client, using data from finished_lines.csv
    
    Args:
    - strategy: BettingStrategy enum value
    - strategy_params: Dictionary of parameters for the selected strategy
    - line_index: Index of the line to simulate from the CSV file
    """
    logging.info(f"Starting simulation with strategy: {strategy.value}")
    if strategy_params:
        logging.info(f"Strategy parameters: {strategy_params}")
    
    # Read the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'finished_lines.csv')
    df = pd.read_csv(csv_path)
    
    if line_index >= len(df):
        logging.error(f"Line index {line_index} is out of range. CSV file has {len(df)} lines.")
        return
        
    # Get the line data
    line = df.iloc[line_index]
    outcomes = list(line['all_outcomes_first_shoe'])
    initial_mode = line['initial_mode']
    
    logging.info(f"Simulating line {line_index} with initial mode {initial_mode}")
    logging.info(f"Outcomes: {outcomes}")
    
    # Initialize the simulator with initial outcomes
    simulator = GameSimulator(outcomes[:6], initial_mode, False, strategy, strategy_params)
    
    # Press the initial mode button
    if initial_mode == 'PPP':
        press_player_start_btn()
    else:
        press_banker_start_btn()
    
    time.sleep(1)  # Wait for the UI to update
    
    try:
        # Go through each outcome in the line
        for outcome in outcomes[6:]:
            # Process the new outcome
            next_bet, new_mode = simulator.simulate_single(new_outcome=outcome)
            logging.info(f"Processing outcome: {outcome}, Next bet will be: {next_bet}, Mode: {new_mode}")
            
            # Create and set the current bet
            current_bet = Bet(next_bet)
            simulator.last_bet = current_bet  # Set the current bet as last_bet
            simulator.bets.append(current_bet)
            
            # Handle the result of the last bet if it exists
            if simulator.last_bet:
                logging.info(f"Last bet was {simulator.last_bet.side}, outcome was {outcome}")
                
                # Process the result
                if outcome == simulator.last_bet.side:
                    logging.info("Result: Win")
                    press_win_btn()
                    simulator.last_bet.set_result('W')
                    simulator.total_pnl += simulator.last_bet.size
                elif outcome == 'T':
                    logging.info("Result: Tie")
                    press_tie_btn()
                    simulator.last_bet.set_result('T')
                else:
                    logging.info("Result: Loss")
                    press_loss_btn()
                    simulator.last_bet.set_result('L')
                    simulator.total_pnl -= simulator.last_bet.size
                
                # Log current PnL
                logging.info(f"Current PnL: {simulator.total_pnl}")
                
                time.sleep(0.5)  # Wait between button presses
            
            time.sleep(0.5)  # Wait between outcomes
            
        # Final statistics
        logging.info("Simulation completed")
        logging.info(f"Final PnL: {simulator.total_pnl}")
        logging.info(f"Total bets: {len(simulator.bets)}")
        win_rate = sum(1 for bet in simulator.bets if bet.result == 'W') / len(simulator.bets) if simulator.bets else 0
        logging.info(f"Win rate: {win_rate:.2%}")
            
    except KeyboardInterrupt:
        logging.info("Simulation stopped by user")
        if simulator:
            logging.info(f"Final PnL: {simulator.total_pnl}")
            logging.info(f"Total bets: {len(simulator.bets)}")
            win_rate = sum(1 for bet in simulator.bets if bet.result == 'W') / len(simulator.bets) if simulator.bets else 0
            logging.info(f"Win rate: {win_rate:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Run a baccarat strategy simulation using historical data')
    parser.add_argument('--strategy', 
                       choices=[s.value for s in BettingStrategy], 
                       default='original',
                       help='The betting strategy to use')
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
    parser.add_argument('--line-index',
                       type=int,
                       default=0,
                       help='Index of the line to simulate from finished_lines.csv')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )
    
    # Convert strategy string to enum
    strategy = BettingStrategy(args.strategy)
    
    # Build strategy parameters based on the strategy type
    strategy_params = {}
    if strategy == BettingStrategy.FOLLOW_STREAK or strategy == BettingStrategy.COUNTER_STREAK:
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
    
    run_simulation(strategy, strategy_params, args.line_index)

if __name__ == "__main__":
    main()