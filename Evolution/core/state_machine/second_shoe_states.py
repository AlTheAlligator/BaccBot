
from datetime import datetime
import logging
import time
from core.nameless import press_banker_only_btn, press_loss_btn, press_player_only_btn, press_player_start_btn, press_win_btn
from core.state_machine.state_machine_base import State


class SecondShoePreparationState(State):
    """Prepares a line for the second shoe with the specific sequence defined in requirements"""
    def __init__(self, name: str, context):
        super().__init__(name, context)
        self.sequence_completed = False
        
    def execute(self):
        logging.info("State: Second Shoe Preparation")
        
        if not self.sequence_completed:
            logging.info("Preparing second shoe line with preset sequence")
            
            # 1. Click "Player Start" mode
            press_player_start_btn(True)
            time.sleep(0.5)
            
            # 2. Click "Win" button to make player only appear
            press_win_btn(True)

            logging.info("Switching to Player Only mode")
            # 2. Select "Player Only"
            press_player_only_btn(True)
            time.sleep(0.5)
            
            logging.info("Executing Win/Loss sequence")
            # 3. Press W and L buttons in the defined sequence
            self._execute_sequence()
            
            # 4. Switch to "Banker Only" mode after the sequence
            press_banker_only_btn(True)
            time.sleep(0.5)
            
            # Set initial virtual drawdown of -770
            self.context.table.bet_manager = None  # Reset bet manager to start fresh
            self.context.table.bet_manager = self.context.create_bet_manager(initial_pnl=-770)
            self.context.table.line_start_time = datetime.now()  # Use datetime.now() for consistency
            
            self.sequence_completed = True
            logging.info("Second shoe preparation complete with virtual drawdown of -770")
            
            # Update the current mode to "PPP" for second shoe betting
            self.context.game.update_mode("PPP")
            
            return "lobby"
        
        return "lobby"
        
    def _execute_sequence(self):
        """Execute the preset Win/Loss button sequence for second shoe preparation"""
        # Sequence from requirements
        sequence = [
            ("W", 3),  # 3x W (First was already hit)
            ("L", 11), # 11x L
            ("W", 2),  # 2x W
            ("L", 2),  # 2x L
            ("W", 1),  # 1x W
            ("L", 1),  # 1x L
            ("W", 2),  # 2x W
            ("L", 1),  # 1x L
            ("W", 1),  # 1x W
            ("L", 3),  # 3x L
            ("W", 3),  # 3x W
            ("L", 1),  # 1x L
            ("W", 1),  # 1x W
            ("L", 1),  # 1x L
            ("W", 1),  # 1x W
            ("L", 11), # 11x L
            ("W", 2),  # 2x W
            ("L", 1),  # 1x L
            ("W", 1),  # 1x W
            ("L", 1),  # 1x L
            ("W", 1),  # 1x W
            ("L", 1),  # 1x L
            ("W", 5),  # 5x W
        ]
        
        for result, count in sequence:
            for _ in range(count):
                if result == "W":
                    press_win_btn()
                else:
                    press_loss_btn()
                time.sleep(0.2)  # Small delay between button presses
            time.sleep(0.2)  # Slightly longer delay between groups