from core.state_machine.state_machine_base import State
import time
import random
import logging
from numpy.random import normal
from core.strategy import play_mode, determine_chip_amount
from core.interaction import random_mouse_placement, random_mouse_move
from core.screencapture import capture_nameless_betbox
from core.ocr import extract_bet_size
from core.evolution import get_bet_allowed, place_bets, shoe_finished

def simulate_decision_time():
    """Quick decision-making simulation"""
    return abs(normal(0.4, 0.1))

class FindBetState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
    
    def execute(self):
        logging.info("State: Finding next bet")
        time.sleep(simulate_decision_time())
        
        next_bet, new_mode = play_mode(
            self.context.game.current_mode,
            self.context.game.outcomes,
            self.context.game.initial_mode,
            self.context.game.last_bet
        )
        
        self.context.game.current_mode = new_mode
        bet_size = extract_bet_size(capture_nameless_betbox())
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
        
        if shoe_finished():
            self.context.game.end_line_reason = "Shoe finished"
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
        chips = determine_chip_amount(self.context.game.current_bet.size)
        place_bets(self.context.game.current_bet.side, chips)
        self.context.game.last_bet = self.context.game.current_bet
        
        time.sleep(random.uniform(0.1, 0.2))
        random_mouse_placement(3000, 1200)
        
        return "wait_result"
