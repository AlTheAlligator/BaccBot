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

# Create a small test dataset with the required columns
outcomes = ['P', 'B', 'P', 'B', 'P', 'B', 'P', 'B', 'P', 'B'] * 10
test_data = pd.DataFrame({
    'all_outcomes_first_shoe': [''.join(outcomes)],  # Join as a single string
    'initial_mode': ['normal'],  # Initial mode
    'four_start': [False]  # Four start flag
})

# Test with HYBRID_FREQUENCY_VOLATILITY strategy
strategy = BettingStrategy.HYBRID_FREQUENCY_VOLATILITY
print(f"Testing simulation with {strategy.value} strategy")

# Get parameter ranges
param_ranges = get_parameter_ranges(strategy)

# Create a simplified parameter set for testing
test_params = {
    'performance_window': 10,
    'min_confidence_diff': 0.05,
    'performance_weight': 0.3,
    'confidence_weight': 0.7,
    'frequency_params': {
        'short_window': 3,
        'medium_window': 10,
        'long_window': 20,
        'min_samples': 3,
        'confidence_threshold': 0.6,
        'pattern_length': 3,
        'banker_bias': 0.01,
        'use_trend_adjustment': True,
        'trend_weight': 0.5,
        'use_pattern_adjustment': True,
        'pattern_weight': 0.5,
        'use_chi_square': False,
        'significance_level': 0.05
    },
    'volatility_params': {
        'short_window': 3,
        'medium_window': 8,
        'long_window': 20,
        'min_samples': 3,
        'high_volatility_threshold': 0.7,
        'low_volatility_threshold': 0.3,
        'confidence_threshold_base': 0.6,
        'confidence_scaling': 0.1,
        'banker_bias': 0.01,
        'use_adaptive_window': True,
        'statistical_mode': 'frequency',
        'pattern_length': 3,
        'min_pattern_occurrences': 3
    }
}

# Run a simple simulation with the test parameters
print("Running simulation with test parameters...")
summary_df, _ = main_sim.simulate_strategies(
    historical_data_df=test_data,
    selected_strategies=[strategy],
    bet_size=1,
    use_optimized_params=False,
    override_params={strategy: [test_params]}
)

# Print the results
if summary_df is not None and not summary_df.empty:
    print("\nSimulation results:")
    print(summary_df)
else:
    print("Simulation failed or returned no results.")
