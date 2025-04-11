import logging
from strategies.betting_strategy import BettingStrategy
from strategies import get_parameter_ranges

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test loading parameter ranges for HYBRID_FREQUENCY_VOLATILITY
strategy = BettingStrategy.HYBRID_FREQUENCY_VOLATILITY
print(f"Testing parameter ranges for {strategy.value}")

# Get parameter ranges
param_ranges = get_parameter_ranges(strategy)

# Check if parameter ranges were loaded successfully
if param_ranges:
    print(f"Successfully loaded parameter ranges for {strategy.value}")
    
    # Print top-level parameters
    print("\nTop-level parameters:")
    for param_name, param_range in param_ranges.items():
        if 'values' in param_range:
            print(f"  {param_name}: {param_range['values']}")
        elif 'min' in param_range and 'max' in param_range:
            print(f"  {param_name}: {param_range['min']} to {param_range['max']} ({param_range['steps']} steps)")
        elif isinstance(param_range, dict):
            print(f"  {param_name}: <nested parameters>")
            
            # Print nested parameters
            print(f"\n  Nested parameters for {param_name}:")
            for nested_param, nested_range in param_range.items():
                if 'values' in nested_range:
                    print(f"    {nested_param}: {nested_range['values']}")
                elif 'min' in nested_range and 'max' in nested_range:
                    print(f"    {nested_param}: {nested_range['min']} to {nested_range['max']} ({nested_range['steps']} steps)")
else:
    print(f"Failed to load parameter ranges for {strategy.value}")
