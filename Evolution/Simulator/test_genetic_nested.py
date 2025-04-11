import main_sim
import random
import pandas as pd
from strategies.betting_strategy import BettingStrategy
from strategies import get_parameter_ranges

print('Testing genetic algorithm with nested parameters...')
test_ranges = {
    'performance_window': {'min': 3, 'max': 6, 'steps': 2},
    'frequency_params': {
        'short_window': {'min': 2, 'max': 4, 'steps': 2},
        'medium_window': {'min': 8, 'max': 10, 'steps': 2}
    }
}
print(f'Test ranges: {test_ranges}')

# Create a mock historical data DataFrame
mock_df = pd.DataFrame({
    'Outcome': ['P', 'B', 'P', 'B', 'P', 'B', 'P', 'B', 'P', 'B'] * 10
})

# Override the _evaluate_fitness_ga function to return a random fitness
original_evaluate_fitness = main_sim._evaluate_fitness_ga

def mock_evaluate_fitness(param_set, strategy, historical_data_df, bet_size):
    # Just return a random fitness value between 0 and 1
    return random.random()

# Replace the function temporarily
main_sim._evaluate_fitness_ga = mock_evaluate_fitness

try:
    # Run the genetic algorithm with the mock fitness function
    population, results = main_sim._run_genetic_algorithm(
        BettingStrategy.ORIGINAL,
        test_ranges,
        {},
        1,
        mock_df,
        population_size=4,
        generations=2,
        mutation_rate=0.1,
        use_parallel=False
    )

    print('\nFinal population:')
    for i, individual in enumerate(population):
        print(f"Individual {i+1}:")
        print(f"  Parameters: {individual}")

    print('\nResults:')
    if isinstance(results, list) and len(results) > 0:
        for i, result in enumerate(results[:5]):
            if isinstance(result, dict):
                # Print key-value pairs for each result
                print(f"Result {i+1}:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Result {i+1}: {result}")

        if len(results) > 5:
            print(f"... and {len(results) - 5} more results")
finally:
    # Restore the original function
    main_sim._evaluate_fitness_ga = original_evaluate_fitness
