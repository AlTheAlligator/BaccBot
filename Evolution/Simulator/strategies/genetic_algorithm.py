"""
Genetic Algorithm strategy implementation.
"""

import logging
import numpy as np
import random
from collections import defaultdict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class GeneticAlgorithmStrategy(BaseStrategy):
    """
    Genetic Algorithm strategy for baccarat betting.

    This strategy evolves a population of betting rules over time, selecting the most
    successful rules and combining them to create new rules that potentially perform better.
    """

    def __init__(self, simulator, params=None):
        """
        Initialize the Genetic Algorithm strategy.

        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)

        # GA parameters
        self.population_size = params.get('population_size', 20)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.crossover_rate = params.get('crossover_rate', 0.7)
        self.generations = params.get('generations', 10)
        self.selection_pressure = params.get('selection_pressure', 2.0)

        # Strategy parameters
        self.min_samples = params.get('min_samples', 20)
        self.confidence_threshold = params.get('confidence_threshold', 0.6)
        self.banker_bias = params.get('banker_bias', 0.01)

        # Initialize population of betting rules
        self.population = self._initialize_population()
        self.best_individual = None

        # For tracking performance
        self.rule_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        self.generation = 0
        self.evolution_interval = 20  # Evolve after this many bets
        self.bet_count = 0

        # Pattern store for rule evaluation
        self.pattern_store = defaultdict(lambda: {'P': 0, 'B': 0})

        logger.info(f"Initialized Genetic Algorithm strategy with population_size={self.population_size}")

    def _initialize_population(self):
        """
        Initialize a population of betting rules.

        Returns:
            list: Population of betting rules
        """
        population = []

        # Create random rules
        for _ in range(self.population_size):
            # Each rule is a dictionary with pattern and action
            rule = {
                'pattern_length': random.randint(2, 5),
                'patterns': {},
                'default_action': random.choice(['P', 'B']),
                'fitness': 0.0
            }

            # Generate some random patterns and actions
            for _ in range(random.randint(3, 10)):
                pattern = tuple(random.choices(['P', 'B'], k=rule['pattern_length']))
                action = random.choice(['P', 'B'])
                confidence = random.uniform(0.5, 1.0)
                rule['patterns'][pattern] = {'action': action, 'confidence': confidence}

            population.append(rule)

        return population

    def _update_pattern_store(self, outcomes):
        """
        Update pattern store with new outcomes.

        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Process patterns of different lengths
        for length in range(2, 6):
            if len(outcomes) <= length:
                continue

            # Process each pattern
            for i in range(len(outcomes) - length):
                pattern = tuple(outcomes[i:i+length])
                next_outcome = outcomes[i+length]

                # Update pattern store
                if next_outcome in ['P', 'B']:
                    self.pattern_store[pattern][next_outcome] += 1

    def _evaluate_rule(self, rule, outcomes):
        """
        Evaluate a rule against historical outcomes.

        Args:
            rule: Betting rule
            outcomes: List of outcomes ('P', 'B')

        Returns:
            float: Fitness score
        """
        if len(outcomes) < rule['pattern_length'] + 1:
            return 0.0

        wins = 0
        losses = 0
        total = 0

        # Evaluate rule on each possible betting opportunity
        for i in range(len(outcomes) - rule['pattern_length']):
            pattern = tuple(outcomes[i:i+rule['pattern_length']])
            actual_outcome = outcomes[i+rule['pattern_length']]

            if pattern in rule['patterns']:
                bet = rule['patterns'][pattern]['action']

                if actual_outcome == 'T':
                    continue  # Skip ties

                total += 1
                if actual_outcome == bet:
                    wins += 1
                else:
                    losses += 1

        # Calculate fitness
        if total == 0:
            return 0.0

        win_rate = wins / total
        coverage = total / (len(outcomes) - rule['pattern_length'])

        # Fitness is a combination of win rate and coverage
        fitness = win_rate * 0.8 + coverage * 0.2

        return fitness

    def _select_parent(self, population):
        """
        Select a parent for reproduction using tournament selection.

        Args:
            population: List of individuals

        Returns:
            dict: Selected parent
        """
        # Tournament selection
        tournament_size = max(2, int(len(population) / 5))
        tournament = random.sample(population, tournament_size)

        # Return the best individual in the tournament
        return max(tournament, key=lambda x: x['fitness'])

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            dict: Child rule
        """
        # Decide whether to perform crossover
        if random.random() > self.crossover_rate:
            return parent1.copy()

        # Create child with properties from both parents
        child = {
            'pattern_length': random.choice([parent1['pattern_length'], parent2['pattern_length']]),
            'patterns': {},
            'default_action': random.choice([parent1['default_action'], parent2['default_action']]),
            'fitness': 0.0
        }

        # Combine patterns from both parents
        all_patterns = set(parent1['patterns'].keys()) | set(parent2['patterns'].keys())
        for pattern in all_patterns:
            if len(pattern) != child['pattern_length']:
                continue

            if pattern in parent1['patterns'] and pattern in parent2['patterns']:
                # Take from either parent with equal probability
                child['patterns'][pattern] = random.choice([
                    parent1['patterns'][pattern],
                    parent2['patterns'][pattern]
                ])
            elif pattern in parent1['patterns']:
                # Take from parent1
                child['patterns'][pattern] = parent1['patterns'][pattern].copy()
            else:
                # Take from parent2
                child['patterns'][pattern] = parent2['patterns'][pattern].copy()

        return child

    def _mutate(self, individual):
        """
        Mutate an individual.

        Args:
            individual: Individual to mutate

        Returns:
            dict: Mutated individual
        """
        # Clone the individual
        mutated = individual.copy()
        mutated['patterns'] = individual['patterns'].copy()

        # Mutate pattern length with low probability
        if random.random() < self.mutation_rate * 0.2:
            mutated['pattern_length'] = random.randint(2, 5)

        # Mutate default action with low probability
        if random.random() < self.mutation_rate * 0.2:
            mutated['default_action'] = 'P' if individual['default_action'] == 'B' else 'B'

        # Mutate patterns
        for pattern in list(mutated['patterns'].keys()):
            # Remove patterns with wrong length
            if len(pattern) != mutated['pattern_length']:
                del mutated['patterns'][pattern]
                continue

            # Mutate action with probability mutation_rate
            if random.random() < self.mutation_rate:
                current_action = mutated['patterns'][pattern]['action']
                mutated['patterns'][pattern]['action'] = 'P' if current_action == 'B' else 'B'

            # Mutate confidence with probability mutation_rate
            if random.random() < self.mutation_rate:
                current_conf = mutated['patterns'][pattern]['confidence']
                # Add some noise to confidence
                new_conf = current_conf + random.uniform(-0.1, 0.1)
                mutated['patterns'][pattern]['confidence'] = max(0.5, min(1.0, new_conf))

        # Add new random patterns with probability mutation_rate
        if random.random() < self.mutation_rate:
            for _ in range(random.randint(1, 3)):
                pattern = tuple(random.choices(['P', 'B'], k=mutated['pattern_length']))
                action = random.choice(['P', 'B'])
                confidence = random.uniform(0.5, 1.0)
                mutated['patterns'][pattern] = {'action': action, 'confidence': confidence}

        return mutated

    def _evolve_population(self, outcomes):
        """
        Evolve the population using genetic operators.

        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        # Evaluate fitness of each individual
        for individual in self.population:
            individual['fitness'] = self._evaluate_rule(individual, outcomes)

        # Sort population by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)

        # Store best individual
        self.best_individual = self.population[0].copy()

        # Create new population
        new_population = []

        # Elitism: keep the best individual
        new_population.append(self.best_individual)

        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._select_parent(self.population)
            parent2 = self._select_parent(self.population)

            # Perform crossover
            child = self._crossover(parent1, parent2)

            # Perform mutation
            child = self._mutate(child)

            # Add to new population
            new_population.append(child)

        # Replace old population
        self.population = new_population
        self.generation += 1

        logger.info(f"Evolved population to generation {self.generation}. Best fitness: {self.best_individual['fitness']:.4f}")

    def _get_best_bet(self, pattern):
        """
        Get the best bet for a pattern based on the best individual.

        Args:
            pattern: Current pattern

        Returns:
            tuple: (bet, confidence)
        """
        if self.best_individual is None:
            return 'B', 0.51  # Default to banker with minimal confidence

        # Check if pattern exists in best individual
        if pattern in self.best_individual['patterns']:
            rule = self.best_individual['patterns'][pattern]
            return rule['action'], rule['confidence']

        # Fall back to default action
        return self.best_individual['default_action'], 0.51

    def get_bet(self, outcomes):
        """
        Determine the next bet using the genetic algorithm.

        Args:
            outcomes: List of outcomes ('P', 'B', 'T')

        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)

        # Filter out ties
        filtered_outcomes = [o for o in outcomes if o in ['P', 'B']]

        # Not enough data - default to Banker (slight edge due to commission)
        if len(filtered_outcomes) < self.min_samples:
            return 'B'

        # Update pattern store
        self._update_pattern_store(filtered_outcomes)

        # Increment bet count
        self.bet_count += 1

        # Evolve population periodically
        if self.bet_count % self.evolution_interval == 0:
            self._evolve_population(filtered_outcomes)

        # If no best individual yet, evolve immediately
        if self.best_individual is None:
            self._evolve_population(filtered_outcomes)

        # Get current pattern
        pattern_length = self.best_individual['pattern_length'] if self.best_individual else 3
        if len(filtered_outcomes) < pattern_length:
            return 'B'  # Default to Banker when not enough history

        current_pattern = tuple(filtered_outcomes[-pattern_length:])

        # Get best bet for current pattern
        bet, confidence = self._get_best_bet(current_pattern)

        # Apply banker bias
        if bet == 'B':
            confidence += self.banker_bias

        # Return the bet (no skipping)
        return bet

    def get_stats(self):
        """Get strategy statistics for debugging."""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_individual['fitness'] if self.best_individual else 0.0,
            "pattern_length": self.best_individual['pattern_length'] if self.best_individual else 0,
            "num_patterns": len(self.best_individual['patterns']) if self.best_individual else 0,
            "bet_count": self.bet_count
        }
