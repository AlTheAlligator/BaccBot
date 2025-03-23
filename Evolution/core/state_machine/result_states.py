from core.state_machine.state_machine_base import State
import logging
import time
import random
from core.interaction import random_mouse_move, click_button
from core.evolution import get_game_result_wrapper
from core.strategy import check_for_end_line
from core.nameless import press_new_line_btn, press_win_btn, press_loss_btn, press_tie_btn, press_end_line_btn
from core.screencapture import get_lobby_btn_coordinates, get_close_running_game_coordinates
from core.discord_manager import on_line_finish

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
        if check_for_end_line(self.context, use_mini_line_exit=False, use_moderate_exit=True):
            logging.info("End line condition met")
            return "end_line"
        return "find_bet"

class EndLineState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Ending line")
        on_line_finish(self.context.get_total_pnl(), self.context.game.end_line_reason, self.context.game.is_second_shoe)
        self.context.export_line_to_csv()
        
        # In second shoe mode or natural shoe finish, don't press end_line_btn
        if not self.context.game.is_second_shoe and self.context.game.end_line_reason != "Shoe finished":
            press_end_line_btn()
            time.sleep(1)
        elif self.context.game.end_line_reason == "Shoe finished":
            self.context.second_shoe_mode()
            logging.info("Shoe finished, entering second shoe mode")
            return "leave_table"
        elif self.context.game.is_second_shoe:
            self.context.game.is_second_shoe = False
            
        press_new_line_btn()
            
        return "leave_table"

class LeaveTableState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)
        
    def execute(self):
        logging.info("State: Leaving table")
        
        # If in second shoe mode, exit the program
        #if self.context.game.is_second_shoe:
        #    logging.info("Second shoe completed, exiting...")
        #    self.context.stop_event.set()
        #    return None
            
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