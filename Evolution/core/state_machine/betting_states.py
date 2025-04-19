from core.strategies.neural_oscillator import NeuralOscillatorStrategy
from core.strategies.chaos_theory import ChaosTheoryStrategy
from core.strategies.momentum_oscillator import MomentumOscillatorStrategy
from core.strategies.volatility_adaptive import VolatilityAdaptiveStrategy
from core.strategies.hybrid_frequency_volatility import HybridFrequencyVolatilityStrategy
from core.state_machine.state_machine_base import State
import time
import random
import logging
from numpy.random import normal
from core.strategy import play_mode, determine_chip_amount
from core.strategies.frequency_analysis import FrequencyAnalysisStrategy
from core.interaction import random_mouse_placement, random_mouse_move
from core.screencapture import capture_nameless_betbox
from core.ocr import extract_bet_size
from core.evolution import get_bet_allowed, place_bets, shoe_finished

def simulate_decision_time():
    """Quick decision-making simulation"""
    return abs(normal(0.1, 0.02))

class FindBetState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        # Initialize frequency analysis strategy if needed
        self.frequency_strategy = None
        if self.context.game.strategy == "frequency_analysis":
            self.frequency_strategy = FrequencyAnalysisStrategy()
        self.hybrid_frequency_volatility_strategy = None
        if self.context.game.strategy == "hybrid_frequency_volatility":
            self.hybrid_frequency_volatility_strategy = HybridFrequencyVolatilityStrategy()
        if self.context.game.strategy == "volatility_adaptive":
            self.volatility_adaptive_strategy = VolatilityAdaptiveStrategy()
        if self.context.game.strategy == "momentum_oscillator":
            self.momentum_oscillator_strategy = MomentumOscillatorStrategy()
        if self.context.game.strategy == "chaos_theory":
            self.chaos_theory_strategy = ChaosTheoryStrategy()
        if self.context.game.strategy == "neural_oscillator":
            self.neural_oscillator_strategy = NeuralOscillatorStrategy()

    def execute(self):
        logging.info("State: Finding next bet")
        time.sleep(simulate_decision_time())

        # Determine which strategy to use
        if self.context.game.strategy == "frequency_analysis":
            # Use frequency analysis strategy
            if self.frequency_strategy is None:
                self.frequency_strategy = FrequencyAnalysisStrategy()

            # Get bet from frequency analysis strategy
            next_bet = self.frequency_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"Frequency analysis strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
                #next_bet, new_mode = play_mode(
                #    self.context.game.current_mode,
                #    self.context.game.outcomes,
                #    self.context.game.initial_mode,
                #    self.context.game.last_bet
                #)
                #self.context.game.current_mode = new_mode
            #else:
                # Keep the current mode when using frequency analysis
                #new_mode = self.context.game.current_mode
        elif self.context.game.strategy == "hybrid_frequency_volatility":
            if self.hybrid_frequency_volatility_strategy is None:
                self.hybrid_frequency_volatility_strategy = HybridFrequencyVolatilityStrategy()

            # Get bet from frequency analysis strategy
            next_bet = self.hybrid_frequency_volatility_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"Hybrid frequency-volatility analysis strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
        elif self.context.game.strategy == "volatility_adaptive":
            if self.volatility_adaptive_strategy is None:
                self.volatility_adaptive_strategy = VolatilityAdaptiveStrategy()

            # Get bet from frequency analysis strategy
            next_bet = self.volatility_adaptive_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"Volatility adaptive analysis strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
        elif self.context.game.strategy == "momentum_oscillator":
            if self.momentum_oscillator_strategy is None:
                self.momentum_oscillator_strategy = MomentumOscillatorStrategy()
            # Get bet from momentum oscillator strategy
            next_bet = self.momentum_oscillator_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"Volatility adaptive analysis strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
        elif self.context.game.strategy == "chaos_theory":
            if self.chaos_theory_strategy is None:
                self.chaos_theory_strategy = ChaosTheoryStrategy()
            # Get bet from chaos theory strategy
            next_bet = self.chaos_theory_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"chaos theory strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
        elif self.context.game.strategy == "neural_oscillator":
            if self.neural_oscillator_strategy is None:
                self.neural_oscillator_strategy = NeuralOscillatorStrategy()
            # Get bet from chaos theory strategy
            next_bet = self.neural_oscillator_strategy.get_bet(self.context.game.outcomes)
            logging.info(f"Neural oscillator strategy recommends: {next_bet}")
            # If strategy returns SKIP, use original strategy as fallback
            if next_bet == "SKIP":
                if self.context.game.initial_mode == "PPP":
                    next_bet = "B"
                else:
                    next_bet = "P"
        else:
            # Use original strategy
            next_bet, new_mode = play_mode(
                self.context.game.current_mode,
                self.context.game.outcomes,
                self.context.game.initial_mode,
                self.context.game.last_bet
            )
            self.context.game.current_mode = new_mode

        bet_size = extract_bet_size(capture_nameless_betbox())
        logging.info(f"Extracted bet size: {bet_size}")
        self.context.game.current_bet = self.context.create_bet(next_bet, bet_size)

        return "wait_bet"

class WaitBetState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        self._last_move_time = 0

    def execute(self):
        logging.info("State: Waiting for bet to be allowed")
        bet_allowed = get_bet_allowed()

        # Mouse movement
        current_time = time.time()
        if current_time - self._last_move_time > 2:
            random_mouse_move(0.003)
            self._last_move_time = current_time

        if shoe_finished() and not self.context.game.is_second_shoe:
            #logging.info("Shoe finished, switching to second shoe mode")
            self.context.game.end_line_reason = "Shoe finished"
            #self.context.second_shoe_mode()
            #return "find_bet"
            return "end_line"

        if bet_allowed:
            return "place_bet"

        time.sleep(0.1)
        return None

class PlaceBetState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)

    def execute(self):
        logging.info("State: Placing bets")

        # Check if we're in test mode
        if self.context.test_mode:
            # Place a minimal bet (10 chip) only every 3rd game
            if self.context.game_counter % 3 == 2:  # 0-indexed, so 2 means 3rd game
                logging.info("Test mode: Placing minimal bet for 3rd game")
                # Always use 10 chip for test mode
                place_bets(self.context.game.current_bet.side, [10])
            else:
                logging.info(f"Test mode: Skipping actual bet for game {self.context.game_counter + 1}")
                # We don't place a bet, but we still need to process it in our tracking
        else:
            # Normal betting mode - determine chips and place bet
            chips = determine_chip_amount(self.context.game.current_bet.size)
            place_bets(self.context.game.current_bet.side, chips)

        # Always record the bet for tracking purposes
        self.context.game.last_bet = self.context.game.current_bet

        time.sleep(random.uniform(0.1, 0.2))
        random_mouse_placement(3000, 1200)

        return "wait_result"
