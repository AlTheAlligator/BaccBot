import asyncio
from transitions.extensions import AsyncMachine
from pyee import EventEmitter
from core.screencapture import capture_history, capture_nameless_betbox
from core.strategy import analyze_first_6, determine_chip_amount, play_mode, check_for_end_line
from core.interaction import random_mouse_placement
from core.nameless import press_win_btn, press_tie_btn, press_loss_btn
from core.evolution import wait_for_game_result, wait_for_bet_allowed
from core.utils import start_keyboard_listener, stop_event
import logging
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
        if self.side == 'B':
            self.pnl = self.size * 0.95 if result == "Win" else -self.size
        else:
            self.pnl = self.size if result == "Win" else -self.size

    def to_dict(self):
        """Convert bet data to a dictionary for CSV export."""
        return {
            "side": self.side,
            "size": self.size,
            "result": self.result,
            "pnl": self.pnl
        }

class BaccaratBot:
    states = ['table_searching', 'open_table', 'waiting_for_bet', 'placing_bet', 'waiting_for_result', 'handling_result']

    def __init__(self):
        # Initialize the state machine
        self.machine = AsyncMachine(model=self, states=BaccaratBot.states, initial='table_searching')
        # State transitions
        self.machine.add_transition('find_table', 'table_searching', 'waiting_for_bet', after='select_table')
        self.machine.add_transition('allow_bet', 'waiting_for_bet', 'placing_bet', conditions='is_betting_allowed')
        self.machine.add_transition('end_line', 'waiting_for_bet', 'table_searching', conditions='should_end_line')
        self.machine.add_transition('place_bet', 'placing_bet', 'waiting_for_result', after='bet')
        self.machine.add_transition('process_result', 'waiting_for_result', 'handling_result', after='handle_result')
        self.machine.add_transition('continue_bet', 'handling_result', 'waiting_for_bet')
        self.machine.add_transition('end_line', 'handling_result', 'table_searching', conditions='should_end_line')

        # Event system
        self.events = EventEmitter()
        self.setup_events()

        # State variables
        self.current_mode = None
        self.last_bet = None
        self.bets = []  # List to store all bets

    def setup_events(self):
        """Setup event listeners."""
        self.events.on('betting_allowed', self.trigger_allow_bet)
        self.events.on('result_available', self.trigger_process_result)
        self.events.on('line_end', self.trigger_end_line)

    # Event Handlers
    def trigger_allow_bet(self):
        """Trigger the 'allow_bet' transition."""
        if self.is_betting_allowed():
            self.allow_bet()

    def trigger_process_result(self):
        """Trigger the 'process_result' transition."""
        self.process_result()

    def trigger_end_line(self):
        """Trigger the 'end_line' transition."""
        self.end_line()

    # State-specific methods
    async def select_table(self):
        print("Searching for a suitable table...")
        await asyncio.sleep(1)  # Simulate delay
        self.current_table = "Table 1"  # Replace with actual logic
        print(f"Selected {self.current_table}.")

    async def is_betting_allowed(self):
        # Replace with actual logic to determine if betting is allowed
        print("Checking if betting is allowed...")
        await asyncio.sleep(1)  # Simulate delay
        return True

    async def should_end_line(self):
        # Replace with logic to determine if the line should end
        
        await asyncio.sleep(1)  # Simulate delay
        return check_for_end_line("BBB")  # Example condition

    async def bet(self):
        print("Placing a bet...")
        side = "Player"  # Example bet side
        size = 100  # Example bet size
        self.bets.append(Bet(side=side, size=size))  # Record the bet
        await asyncio.sleep(1)  # Simulate delay

    async def handle_result(self):
        print("Handling result...")
        result = "Win"  # Example result
        if self.bets:
            self.bets[-1].set_result(result)  # Set the result of the last bet

    async def play_loop(self):
        """Main loop for the Baccarat bot."""
        while True:
            if self.state == 'table_searching':
                await self.find_table()

            elif self.state == 'waiting_for_bet':
                if await self.should_end_line():
                    self.end_line()
                else:
                    self.events.emit('betting_allowed')

            elif self.state == 'placing_bet':
                await self.place_bet()

            elif self.state == 'waiting_for_result':
                self.events.emit('result_available')

            elif self.state == 'handling_result':
                if await self.should_end_line():
                    self.end_line()
                else:
                    self.continue_bet()

    def export_bets_to_csv(self, file_path):
        """Export all bets to a CSV file."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['side', 'size', 'result', 'pnl'])
            writer.writeheader()
            for bet in self.bets:
                writer.writerow(bet.to_dict())
        print(f"Bets exported to {file_path}")


if __name__ == "__main__":
    bot = BaccaratBot()
    try:
        asyncio.run(bot.play_loop())
    except KeyboardInterrupt:
        bot.export_bets_to_csv('bets.csv')
        print("Program terminated. Bets saved to bets.csv.")
