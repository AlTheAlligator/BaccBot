# The main entry point combines all the modules.

from core.state_detection import detect_game_state
from core.strategy import analyze_first_6, determine_chip_amount, play_mode, check_for_end_line
from core.interaction import click_button, random_mouse_placement
from core.screencapture import capture_history, capture_nameless_window, capture_nameless_cubes, capture_nameless_betbox, capture_game_result, capture_bet_allowed
from core.ocr import extract_cubes_and_numbers, extract_bet_size
from core.nameless import press_win_btn, press_tie_btn, press_loss_btn, press_player_only_btn, press_banker_only_btn, press_new_line_btn, press_end_line_btn, press_reduce_btn
from core.evolution import wait_for_game_result, wait_for_bet_allowed, place_bets
from core.utils import start_keyboard_listener, stop_event
import logging
import time
import random
import csv

logging.basicConfig(level=logging.INFO)

class Bet:
    """Class to represent a single bet."""
    def __init__(self, side, size, result=None, pnl=0):
        self.side = side  # "Player" or "Banker"
        self.size = size  # Amount bet
        self.result = result  # "Win", "Loss", or "Tie"
        self.pnl = pnl  # Profit or loss from the bet

    def set_result(self, result):
        """Set the result of the bet."""
        self.result = result
        if result == "T":
            self.pnl = 0
        elif result == "L":
            self.pnl = -self.size
        else:
            if self.side == 'B':
                self.pnl = self.size * 0.95
            else:
                self.pnl = self.size

    def to_dict(self):
        """Convert bet data to a dictionary for CSV export."""
        return {
            "side": self.side,
            "size": self.size,
            "result": self.result,
            "pnl": self.pnl
        }
    
class BaccaratBot:
    def __init__(self):
        self.bets = []
        self.current_mode = None
        self.last_bet = None
        self.initial_mode = None

    def export_bets_to_csv(self, file_path):
        """Export all bets to a CSV file."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['side', 'size', 'result', 'pnl'])
            writer.writeheader()
            for bet in self.bets:
                writer.writerow(bet.to_dict())
        print(f"Bets exported to {file_path}")

    def wait_for_first_6_games_analysis(self):
        while not stop_event.is_set():
            logging.info("Checking game history...")
            outcomes = capture_history(True)
            action = analyze_first_6(outcomes)
            
            if action == "Skip":
                logging.info("Skipping the shoe.")
                return action
            elif action == "Analyzing":
                logging.info("Analyzing the first 6 games...")
                time.sleep(1)
                continue
            else:
                logging.info("Starting the game.")
                return action
            
    def get_total_pnl(self):
        return sum([bet.pnl for bet in self.bets])

    def play_game(self, initial_mode):
        self.initial_mode = initial_mode
        self.current_mode = initial_mode
        self.last_bet = None
        while not stop_event.is_set():
            # Capture game history
            logging.info("Analyzing current game history...")
            outcomes = capture_history(True)

            # Determine bet size
            bet_size = extract_bet_size(capture_nameless_betbox())
            logging.info(f"Detected bet size: {bet_size}")
            chips = determine_chip_amount(bet_size)
            logging.info(f"Chips to bet: {chips}")

            # Wait for betting to be allowed
            if wait_for_bet_allowed() and (not stop_event.is_set()):
                logging.info("Betting is allowed. Placing bet...")
                next_bet, self.current_mode = play_mode(self.current_mode, outcomes, self.initial_mode, self.last_bet)
                logging.info(f"Next bet: {next_bet}")
                logging.info(f"Current mode: {self.current_mode}")
                logging.info(f"Chips to bet: {chips}")
                place_bets(next_bet, chips)
                self.bets.append(Bet(next_bet, bet_size))
                self.last_bet = next_bet
                time.sleep(random.uniform(0.08, 0.5))  # Adjust delay for betting timing
                random_mouse_placement(1500, 1080)

            # Wait for game result
            game_result = wait_for_game_result()
            if game_result == next_bet:
                logging.info("Win!")
                press_win_btn()
                self.bets[-1].set_result("W")
            elif game_result == "T":
                logging.info("Tie!")
                press_tie_btn()
                self.bets[-1].set_result("T")
            else:
                logging.info("Loss!")
                press_loss_btn()
                self.bets[-1].set_result("L")

            # Check for end line
            if check_for_end_line(self.initial_mode, self.get_total_pnl(), 3, 200):
                logging.info("Ending the line.")
                press_end_line_btn()
                return
        


def main():
    print("Starting Baccarat Automation...")

    # Start the keyboard listener
    start_keyboard_listener()

    # Wait for the first 6 games to be analyzed
    bot = BaccaratBot()
    action = bot.wait_for_first_6_games_analysis()
    if action == "Skip":
        return
    else:
        try:
            bot.play_game(action)
        except KeyboardInterrupt:
            bot.export_bets_to_csv('bets.csv')
            print("Program terminated. Bets saved to bets.csv.")

if __name__ == "__main__":
    main()
    