"""
Test script for the nameless betting system Line class.
"""

import logging
import random
from line import Line, LineManager, simulate_line

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_basic_line():
    """Test basic line functionality with a simple sequence of outcomes."""
    print("\n=== Basic Line Test ===")
    
    # Create a line with initial side 'B' and default cubes for both lists
    line = Line('B')
    print(f"Initial line state: {line}")
    
    # Define a sequence of outcomes
    outcomes = ['B', 'P', 'B', 'B', 'P', 'T', 'B', 'P', 'P']
    
    # Process each outcome
    for i, outcome in enumerate(outcomes):
        # Place a bet (always on the initial side for this test)
        bet = line.place_bet(line.initial_side)
        if bet:
            print(f"\nOutcome {i+1}: {outcome}")
            print(f"Placed bet: {bet}")
            
            # Process the outcome
            result = line.process_outcome(outcome)
            
            # Print the result
            print(f"Result: {bet.result}, Profit: {bet.profit:.2f}")
            print(f"Left cubes: {result['left_cubes']}, Right cubes: {result['right_cubes']}")
            print(f"Total PnL: {result['total_pnl']:.2f}, Active: {result['is_active']}")
        else:
            print(f"\nOutcome {i+1}: {outcome} - Line is no longer active")
            break
    
    # Print final state
    print("\nFinal line state:")
    print(f"Left cubes: {line.left_cubes}, Right cubes: {line.right_cubes}")
    print(f"Total PnL: {line.pnl:.2f}, Active: {line.is_active}")
    print(f"Total bets: {len(line.bets)}, Total outcomes: {len(line.outcomes)}")

def test_line_manager():
    """Test the LineManager with multiple lines."""
    print("\n=== Line Manager Test ===")
    
    # Create a line manager
    manager = LineManager()
    
    # Create 3 lines with different initial sides and cube configurations
    manager.create_line('B')
    manager.create_line('P')
    manager.create_line('B')
    
    # Process some outcomes for each line
    for i in range(3):
        line = manager.lines[i]
        print(f"\nProcessing line {i+1}: {line}")
        
        # Generate random outcomes
        outcomes = random.choices(['P', 'B', 'T'], weights=[0.45, 0.46, 0.09], k=10)
        
        # Process each outcome
        for j, outcome in enumerate(outcomes):
            # Place a bet (always on the initial side for this test)
            bet = line.place_bet(line.initial_side)
            if bet:
                result = line.process_outcome(outcome)
                print(f"Outcome {j+1}: {outcome}, Result: {bet.result}, Profit: {bet.profit:.2f}")
                print(f"Left cubes: {result['left_cubes']}, Right cubes: {result['right_cubes']}")
            else:
                print(f"Outcome {j+1}: {outcome} - Line is no longer active")
                break
        
        print(f"Final state: {line}")
    
    # Print overall statistics
    stats = manager.get_statistics()
    print("\nOverall Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

def test_simulation():
    """Test the simulation function with a strategy."""
    print("\n=== Simulation Test ===")
    
    # Define a simple strategy function that alternates between P and B
    def alternating_strategy(outcomes, line):
        if not outcomes:
            return line.initial_side
        last_bet = line.current_side
        return 'P' if last_bet == 'B' else 'B'
    
    # Generate random outcomes
    outcomes = random.choices(['P', 'B', 'T'], weights=[0.45, 0.45, 0.1], k=20)
    
    # Run the simulation
    line, stats = simulate_line(
        outcomes=outcomes,
        initial_side='B',
        left_cubes=[1, 2],
        right_cubes=[1, 3],
        strategy_func=alternating_strategy
    )
    
    # Print the outcomes
    print(f"Outcomes: {', '.join(outcomes)}")
    
    # Print the simulation results
    print("\nSimulation Results:")
    print(f"Final state: {line}")
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_basic_line()
    test_line_manager()
    test_simulation()
