"""
Evolutionary Game Theory strategy implementation.
"""

import logging
import numpy as np
import random
from collections import defaultdict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class EvolutionaryGameTheoryStrategy(BaseStrategy):
    """
    Evolutionary Game Theory strategy that models the game as a population of strategies.
    
    This strategy applies concepts from evolutionary game theory to model the dealer
    and other players as agents with evolving strategies. It adapts its own strategy
    to maximize expected payoff against the evolving population.
    """
    
    def __init__(self, simulator, params=None):
        """
        Initialize the Evolutionary Game Theory strategy.
        
        Args:
            simulator: The game simulator instance
            params: Dictionary of parameters for the strategy
        """
        super().__init__(simulator, params)
        
        # Strategy parameters
        self.population_size = params.get('population_size', 10)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.selection_pressure = params.get('selection_pressure', 1.5)
        self.min_samples = params.get('min_samples', 20)
        self.banker_bias = params.get('banker_bias', 0.01)
        self.evolution_interval = params.get('evolution_interval', 10)
        
        # Initialize strategy population
        self.strategies = self._initialize_strategies()
        self.strategy_fitness = {i: 1.0 for i in range(self.population_size)}
        
        # For tracking state
        self.generation = 0
        self.bet_count = 0
        self.last_bets = {}
        self.payoff_matrix = self._initialize_payoff_matrix()
        
        # Pattern store for strategy evaluation
        self.pattern_store = defaultdict(lambda: {'P': 0, 'B': 0})
        
        logger.info(f"Initialized Evolutionary Game Theory strategy with population_size={self.population_size}")
    
    def _initialize_strategies(self):
        """
        Initialize a population of strategies.
        
        Returns:
            list: Population of strategies
        """
        strategies = []
        
        # Strategy types
        strategy_types = [
            'frequency',  # Bet on most frequent outcome
            'streak',     # Bet on current streak
            'pattern',    # Bet based on pattern matching
            'alternating',  # Bet on alternating pattern
            'contrarian',   # Bet against recent trend
            'random'        # Random betting
        ]
        
        # Create strategies
        for _ in range(self.population_size):
            strategy = {
                'type': random.choice(strategy_types),
                'params': {
                    'window_size': random.randint(5, 30),
                    'pattern_length': random.randint(2, 5),
                    'threshold': random.uniform(0.5, 0.7),
                    'banker_bias': random.uniform(0, 0.05)
                }
            }
            strategies.append(strategy)
        
        return strategies
    
    def _initialize_payoff_matrix(self):
        """
        Initialize the payoff matrix for strategy interactions.
        
        Returns:
            dict: Payoff matrix
        """
        payoff_matrix = {}
        
        # Strategy types
        strategy_types = [
            'frequency',
            'streak',
            'pattern',
            'alternating',
            'contrarian',
            'random'
        ]
        
        # Initialize with equal payoffs
        for s1 in strategy_types:
            payoff_matrix[s1] = {}
            for s2 in strategy_types:
                payoff_matrix[s1][s2] = 0.0
        
        return payoff_matrix
    
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
                self.pattern_store[pattern][next_outcome] += 1
    
    def _update_fitness(self, outcomes):
        """
        Update strategy fitness based on recent performance.
        
        Args:
            outcomes: List of outcomes ('P', 'B')
        """
        if len(outcomes) < 2 or len(self.last_bets) == 0:
            return
        
        # Get most recent outcome
        last_outcome = outcomes[-1]
        
        # Update fitness for each strategy
        for i, strategy in enumerate(self.strategies):
            if i in self.last_bets:
                bet = self.last_bets[i]
                
                # Calculate reward
                if last_outcome == bet:
                    reward = 1.0 if bet == 'P' else 0.95  # Banker commission
                else:
                    reward = -1.0
                
                # Update fitness with exponential moving average
                alpha = 0.1  # Learning rate
                self.strategy_fitness[i] = (1 - alpha) * self.strategy_fitness[i] + alpha * reward
                
                # Update payoff matrix
                s_type = strategy['type']
                for j, other in enumerate(self.strategies):
                    if j in self.last_bets:
                        o_type = other['type']
                        o_bet = self.last_bets[j]
                        
                        # If strategies agree, they share reward
                        if bet == o_bet:
                            self.payoff_matrix[s_type][o_type] = (1 - alpha) * self.payoff_matrix[s_type][o_type] + alpha * reward
        
        # Clear last bets
        self.last_bets = {}
    
    def _evolve_strategies(self):
        """
        Evolve the strategy population using genetic operators.
        """
        # Sort strategies by fitness
        sorted_indices = sorted(range(len(self.strategies)), key=lambda i: self.strategy_fitness[i], reverse=True)
        sorted_strategies = [self.strategies[i] for i in sorted_indices]
        sorted_fitness = [self.strategy_fitness[i] for i in sorted_indices]
        
        # Create new population
        new_strategies = []
        new_fitness = {}
        
        # Elitism: keep the best strategy
        new_strategies.append(sorted_strategies[0].copy())
        new_fitness[0] = sorted_fitness[0]
        
        # Generate rest of population through selection, crossover, and mutation
        for i in range(1, self.population_size):
            # Selection (tournament selection)
            parent1 = self._select_parent(sorted_strategies, sorted_fitness)
            parent2 = self._select_parent(sorted_strategies, sorted_fitness)
            
            # Crossover
            if random.random() < 0.7:  # Crossover rate
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self._mutate(child)
            
            # Add to new population
            new_strategies.append(child)
            new_fitness[i] = 1.0  # Reset fitness for new strategies
        
        # Replace old population
        self.strategies = new_strategies
        self.strategy_fitness = new_fitness
        self.generation += 1
        
        logger.info(f"Evolved population to generation {self.generation}")
    
    def _select_parent(self, strategies, fitness):
        """
        Select a parent for reproduction using tournament selection.
        
        Args:
            strategies: List of strategies
            fitness: List of fitness values
            
        Returns:
            dict: Selected parent strategy
        """
        # Tournament selection
        tournament_size = max(2, self.population_size // 5)
        tournament_indices = random.sample(range(len(strategies)), tournament_size)
        
        # Return the best strategy in the tournament
        best_idx = max(tournament_indices, key=lambda i: fitness[i])
        return strategies[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parent strategies.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            dict: Child strategy
        """
        child = {
            'type': random.choice([parent1['type'], parent2['type']]),
            'params': {}
        }
        
        # Mix parameters
        for param in set(parent1['params'].keys()) | set(parent2['params'].keys()):
            if param in parent1['params'] and param in parent2['params']:
                # Take from either parent with equal probability
                child['params'][param] = random.choice([parent1['params'][param], parent2['params'][param]])
            elif param in parent1['params']:
                child['params'][param] = parent1['params'][param]
            else:
                child['params'][param] = parent2['params'][param]
        
        return child
    
    def _mutate(self, strategy):
        """
        Mutate a strategy.
        
        Args:
            strategy: Strategy to mutate
            
        Returns:
            dict: Mutated strategy
        """
        mutated = strategy.copy()
        mutated['params'] = strategy['params'].copy()
        
        # Mutate type with low probability
        if random.random() < self.mutation_rate * 0.2:
            strategy_types = ['frequency', 'streak', 'pattern', 'alternating', 'contrarian', 'random']
            mutated['type'] = random.choice(strategy_types)
        
        # Mutate parameters
        for param, value in mutated['params'].items():
            if random.random() < self.mutation_rate:
                if param == 'window_size':
                    mutated['params'][param] = max(5, min(30, value + random.randint(-5, 5)))
                elif param == 'pattern_length':
                    mutated['params'][param] = max(2, min(5, value + random.randint(-1, 1)))
                elif param == 'threshold':
                    mutated['params'][param] = max(0.5, min(0.7, value + random.uniform(-0.05, 0.05)))
                elif param == 'banker_bias':
                    mutated['params'][param] = max(0, min(0.05, value + random.uniform(-0.01, 0.01)))
        
        return mutated
    
    def _get_strategy_bet(self, strategy, outcomes):
        """
        Get bet from a specific strategy.
        
        Args:
            strategy: Strategy to use
            outcomes: List of outcomes ('P', 'B')
            
        Returns:
            str: 'P' or 'B'
        """
        s_type = strategy['type']
        params = strategy['params']
        
        if s_type == 'frequency':
            # Bet on most frequent outcome in window
            window_size = params.get('window_size', 20)
            window = outcomes[-window_size:] if len(outcomes) >= window_size else outcomes
            
            p_count = window.count('P')
            b_count = window.count('B')
            
            # Apply banker bias
            b_count += b_count * params.get('banker_bias', 0.01)
            
            return 'P' if p_count > b_count else 'B'
            
        elif s_type == 'streak':
            # Bet on current streak if long enough
            if len(outcomes) < 3:
                return 'B'
                
            current = outcomes[-1]
            streak_length = 1
            for i in range(len(outcomes) - 2, -1, -1):
                if outcomes[i] == current:
                    streak_length += 1
                else:
                    break
            
            threshold = params.get('threshold', 0.6)
            if streak_length >= 3 or random.random() < threshold:
                return current
            else:
                return 'P' if current == 'B' else 'B'
                
        elif s_type == 'pattern':
            # Bet based on pattern matching
            pattern_length = params.get('pattern_length', 3)
            
            if len(outcomes) < pattern_length:
                return 'B'
                
            pattern = tuple(outcomes[-pattern_length:])
            
            if pattern in self.pattern_store:
                counts = self.pattern_store[pattern]
                p_count = counts['P']
                b_count = counts['B']
                
                # Apply banker bias
                b_count += b_count * params.get('banker_bias', 0.01)
                
                if p_count + b_count > 0:
                    return 'P' if p_count > b_count else 'B'
            
            return 'B'
            
        elif s_type == 'alternating':
            # Bet on alternating pattern
            if len(outcomes) < 2:
                return 'B'
                
            last = outcomes[-1]
            return 'P' if last == 'B' else 'B'
            
        elif s_type == 'contrarian':
            # Bet against recent trend
            window_size = params.get('window_size', 10)
            window = outcomes[-window_size:] if len(outcomes) >= window_size else outcomes
            
            p_count = window.count('P')
            b_count = window.count('B')
            
            # Bet against the trend
            return 'B' if p_count > b_count else 'P'
            
        else:  # random
            # Random betting with slight banker bias
            return 'B' if random.random() < 0.51 else 'P'
    
    def get_bet(self, outcomes):
        """
        Determine the next bet using evolutionary game theory.
        
        Args:
            outcomes: List of outcomes ('P', 'B', 'T')
            
        Returns:
            str: 'P' or 'B'
        """
        # Validate outcomes
        self._validate_outcome_list(outcomes)
        
        # Always start from game 7
        if len(outcomes) < 7:
            return 'B'  # Default to Banker for initial games
        
        # Filter out ties
        filtered = [o for o in outcomes if o in ['P', 'B']]
        
        # Not enough data - use simple frequency analysis
        if len(filtered) < self.min_samples:
            p_count = filtered.count('P')
            b_count = filtered.count('B')
            
            # Apply banker bias
            b_count += b_count * self.banker_bias
            
            # Return the more frequent outcome
            return 'P' if p_count > b_count else 'B'
        
        # Update pattern store
        self._update_pattern_store(filtered)
        
        # Update fitness based on recent performance
        self._update_fitness(filtered)
        
        # Increment bet count
        self.bet_count += 1
        
        # Evolve strategies periodically
        if self.bet_count % self.evolution_interval == 0:
            self._evolve_strategies()
        
        # Get votes from each strategy in the population
        votes = {'P': 0, 'B': 0}
        for i, strategy in enumerate(self.strategies):
            bet = self._get_strategy_bet(strategy, filtered)
            self.last_bets[i] = bet  # Store for fitness update
            votes[bet] += self.strategy_fitness[i]
        
        # Apply banker bias
        votes['B'] += self.banker_bias
        
        # Return the bet with the highest weighted votes
        return 'P' if votes['P'] > votes['B'] else 'B'
    
    def get_stats(self):
        """Get strategy statistics for debugging."""
        # Count strategy types
        type_counts = {}
        for strategy in self.strategies:
            s_type = strategy['type']
            type_counts[s_type] = type_counts.get(s_type, 0) + 1
        
        # Calculate average fitness
        avg_fitness = sum(self.strategy_fitness.values()) / len(self.strategy_fitness)
        
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "strategy_types": type_counts,
            "average_fitness": avg_fitness,
            "pattern_count": len(self.pattern_store),
            "bet_count": self.bet_count
        }
