from core.state_machine.state_machine_base import State
import logging
import time
import random
from core.interaction import random_mouse_move, click_button
from core.evolution import get_game_result_wrapper
from core.strategy import check_for_end_line
from core.nameless import press_new_line_btn, press_win_btn, press_loss_btn, press_tie_btn, press_end_line_btn
from core.screencapture import get_lobby_btn_coordinates, get_close_running_game_coordinates, capture_nameless_cubes
from core.discord_manager import on_line_finish
from core.ocr import extract_cubes_and_numbers
from core.gsheets_handler import write_result_line

class WaitResultState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        self._last_move_time = 0
        
    def execute(self):
        logging.debug("State: Waiting for game result")
        game_result = get_game_result_wrapper()
        
        current_time = time.time()
        if current_time - self._last_move_time > 2:
            random_mouse_move(0.001)
            self._last_move_time = current_time
        
        if game_result != "Waiting for Results":
            self.context.game.game_result = game_result
            return "handle_result"
            
        time.sleep(0.1)
        return None

class HandleResultState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Handling game result")
        game_result = self.context.game.game_result
            
        # Add the new result to our stored outcomes
        self.context.game.outcomes.append(game_result)
        
        # Increment game counter in test mode
        if self.context.test_mode:
            self.context.game_counter += 1
            logging.info(f"Test mode: Game count {self.context.game_counter}")
        
        if game_result == self.context.game.current_bet.side:
            logging.info("Result: Win")
            press_win_btn()
            self.context.game.current_bet.set_result("W")
        elif game_result == "T":
            logging.info("Result: Tie")
            press_tie_btn()
            self.context.game.current_bet.set_result("T")
        else:
            logging.info("Result: Loss")
            press_loss_btn()
            self.context.game.current_bet.set_result("L")
            
        return "check_end"

class CheckEndState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Checking end conditions")
        logging.info(f"PNL: {self.context.get_total_pnl()} with {len(self.context.table.bet_manager.get_all_bets())} bets")
        
        # Use the check_for_end_line function which now properly handles second shoe exit conditions in strategy.py
        if check_for_end_line(self.context, use_mini_line_exit=False, use_moderate_exit=True):
            logging.info(f"End line condition met: {self.context.game.end_line_reason}")
            return "end_line"
            
        return "find_bet"

class EndLineState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Ending line")
        # Store cube count before ending the line if the shoe is finished
        if self.context.game.end_line_reason == "Shoe finished":
            cube_count, extracted_numbers = extract_cubes_and_numbers(capture_nameless_cubes())
            self.context.game.cube_count = cube_count
            self.context.game.cube_values = extracted_numbers
            logging.info(f"Stored cube count before ending line: {cube_count}, values: {extracted_numbers}")
            # Store first shoe drawdown for reference
            self.context.game.first_shoe_drawdown = self.context.get_total_pnl()
        
        on_line_finish(self.context.get_total_pnl(), self.context.game.end_line_reason, self.context.game.is_second_shoe)
        self.context.export_line_to_csv()
        
        # Write results to Google Sheets
        write_result_line(self.context)
        
        # In natural shoe finish, start a new shoe instead of going to second shoe mode
        if self.context.game.end_line_reason != "Shoe finished":
            press_end_line_btn()
            time.sleep(1)
        elif self.context.game.end_line_reason == "Shoe finished":
            # Start a new shoe instead of entering second shoe mode
            logging.info("Shoe finished, starting a new shoe")
            # Reset for new shoe
            self.context.reset_table()
            
        press_new_line_btn()
            
        return "leave_table"

class LeaveTableState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Leaving table")
        
        # If in second shoe mode, exit the program
        if self.context.game.is_second_shoe and self.context.game.end_line_reason != "":
            logging.info("Second shoe completed, exiting...")
            self.context.stop_event.set()
            return None
            
        click_button(get_lobby_btn_coordinates())
        time.sleep(random.uniform(3, 4))
        click_button(get_close_running_game_coordinates())
        return "lobby"

class WaitNextGameState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        self._last_move_time = 0
        
    def execute(self):
        logging.debug("State: Waiting for next game")
        game_result = get_game_result_wrapper()
        
        current_time = time.time()
        if current_time - self._last_move_time > 2:
            random_mouse_move(0.001)
            self._last_move_time = current_time
        
        if game_result != "Waiting for Results":
            # Always append outcomes as we need them for strategy
            self.context.game.outcomes.append(game_result)
            return "find_bet"
            
        time.sleep(0.1)
        return None