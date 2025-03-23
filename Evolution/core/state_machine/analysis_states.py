from core.state_machine.state_machine_base import State
import logging
import time
import random
from core.nameless import (
    press_tie_btn, press_banker_start_btn, press_player_start_btn, 
    press_banker_only_btn, press_player_only_btn
)
from core.strategy import analyze_first_6, get_outcomes_without_not_played
from core.screencapture import capture_history

class InitialAnalysisState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        self.consecutive_checks = 0
        
    def execute(self):
        logging.info("State: Initial Analysis")
        
        # Initialize outcomes if empty
        if not self.context.game.outcomes:
            raw_outcomes = capture_history(True)
            self.context.game.outcomes = get_outcomes_without_not_played(raw_outcomes)
        
        action = analyze_first_6(self.context.game.outcomes, self.context.game.bias, 0)
        
        if action == "Skip":
            logging.info("Skipping the shoe.")
            time.sleep(random.uniform(0.2, 0.3))
            self.context.table.add_to_cooldown(self.context.table.coordinates)
            return "leave_table"
        elif action == "Analyzing":
            time.sleep(1)
            return None  # Stay in analysis state
        else:
            logging.info("Analysis complete, starting game.")
            self.context.game.update_mode(action)
            # If second shoe, go straight to finding bets
            if self.context.game.is_second_shoe:
                logging.info("Second shoe mode")
                return "wait_next_game"  # Changed to wait_next_game instead of find_bet to catch next game result
            return "initialize_line"

class InitializeLineState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Initialize Line")
        if self.context.game.initial_mode == "PPP":
            press_banker_start_btn(True)
            press_tie_btn(True)
            press_banker_only_btn(True)
        else:
            press_player_start_btn(True)
            press_tie_btn(True)
            press_player_only_btn(True)
        
        return "wait_next_game"