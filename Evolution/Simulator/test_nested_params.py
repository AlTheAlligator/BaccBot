import main_sim
from strategies.betting_strategy import BettingStrategy
from strategies import get_parameter_ranges

print('Testing parameter combinations for nested parameters...')
test_ranges = {
    'performance_window': {'min': 3, 'max': 6, 'steps': 2},
    'frequency_params': {
        'short_window': {'min': 2, 'max': 4, 'steps': 2},
        'medium_window': {'min': 8, 'max': 10, 'steps': 2}
    }
}
print(f'Test ranges: {test_ranges}')
param_sets = main_sim.generate_parameter_combinations({}, test_ranges)
print(f'Generated {len(param_sets)} parameter combinations')
print('Parameter sets:')
for i, params in enumerate(param_sets):
    print(f'Set {i+1}: {params}')
