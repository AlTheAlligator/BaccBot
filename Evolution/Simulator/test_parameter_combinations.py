import logging
import pandas as pd
import main_sim
from strategies.betting_strategy import BettingStrategy
from strategies import get_parameter_ranges

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test with HYBRID_FREQUENCY_VOLATILITY strategy
strategy = BettingStrategy.HYBRID_FREQUENCY_VOLATILITY
print(f"Testing parameter combinations for {strategy.value} strategy")

# Create a small test dataset with the required columns
outcomes = ['P', 'B', 'P', 'B', 'P', 'B', 'P', 'B', 'P', 'B'] * 10
test_data = pd.DataFrame({
    'timestamp': ['2023-01-01 00:00:00'],
    'all_outcomes_first_shoe': [''.join(outcomes)],  # Join as a single string
    'initial_mode': ['normal'],  # Initial mode
    'four_start': [False]  # Four start flag
})

# Test parameter combinations
print("Testing parameter combinations...")
best_params, results = main_sim.test_parameter_combinations(
    strategy=strategy,
    bet_size=1,
    use_parallel=False,
    use_genetic=False,
    base_params={}
)

# Print the results
if best_params is not None:
    print("\nBest parameters found:")
    for metric, params in best_params.items():
        print(f"Best for {metric}:")
        print(params)
else:
    print("No best parameters found.")
