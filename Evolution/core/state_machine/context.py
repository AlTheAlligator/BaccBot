from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime
from core.gsheets_handler import write_result_line
from core.bet_manager import BetManager, Bet
from threading import Event
import logging
import csv
import os
from core.strategy import get_first_6_non_ties

@dataclass
class GameContext:
    """Context object containing game-specific state"""
    initial_mode: str = ""
    current_mode: str = ""
    current_bet: Optional[Bet] = None
    last_bet: Optional[Bet] = None
    game_result: str = ""
    end_line_reason: str = ""
    bias: str = ""
    outcomes: List[str] = field(default_factory=list)
    first_shoe_outcomes: List[str] = field(default_factory=list)
    first_shoe_drawdown: float = 0.0
    is_second_shoe: bool = False
    cube_count: int = 0
    cube_values: List[int] = field(default_factory=list)
    strategy: str = "original"  # Strategy to use: "original" or "frequency_analysis"

    def update_mode(self, new_mode: str):
        """Updates the current mode and initial mode if not set"""
        if not self.initial_mode:
            self.initial_mode = new_mode
        self.current_mode = new_mode

@dataclass
class TableContext:
    """Context object containing table-specific state"""
    coordinates: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]] = (None, None, None, None)
    cooldown: List[Tuple[tuple, float]] = field(default_factory=list)
    bet_manager: Optional[BetManager] = None
    line_start_time: Optional[datetime] = None

    def get_bet_stats(self) -> Tuple[int, int, int]:
        """Get win/loss/tie counts from current line"""
        if not self.bet_manager:
            return (0, 0, 0)
        bets = self.bet_manager.get_all_bets()
        wins = sum(1 for bet in bets if bet.result == "W")
        losses = sum(1 for bet in bets if bet.result == "L")
        ties = sum(1 for bet in bets if bet.result == "T")
        return (wins, losses, ties)

    def add_to_cooldown(self, coordinates: tuple):
        """Add table coordinates to cooldown list"""
        self.cooldown.append((coordinates, datetime.now().timestamp()))

    def is_in_cooldown(self, coordinates: tuple) -> bool:
        """Check if table coordinates are in cooldown"""
        return coordinates in [c[0] for c in self.cooldown]

class StateMachineContext:
    """Main context object that holds all state machine context"""
    def __init__(self, stop_event: Event, is_second_shoe: bool = False, initial_drawdown: float = None, test_mode: bool = False, strategy: str = "original"):
        self.stop_event = stop_event
        self.game = GameContext(is_second_shoe=is_second_shoe, strategy=strategy)
        self.table = TableContext()
        self.test_mode = test_mode
        self.game_counter = 0

        # Initialize bet manager with initial drawdown for second shoe
        if is_second_shoe and initial_drawdown is not None:
            self.table.bet_manager = self.create_bet_manager(initial_pnl=initial_drawdown)
            self.table.line_start_time = datetime.now()

    def create_bet_manager(self, initial_pnl: float = 0.0) -> BetManager:
        """Creates a new bet manager with an optional initial PnL value"""
        return BetManager(initial_pnl=initial_pnl)

    def create_bet(self, side: str, size: float) -> Bet:
        """Creates a new bet and adds it to the bet manager"""
        if self.table.bet_manager is None:
            self.table.bet_manager = self.create_bet_manager()
            self.table.line_start_time = datetime.now()
        bet = Bet(side, size)
        self.table.bet_manager.add_bet(bet)
        return bet

    def get_total_pnl(self) -> float:
        """Gets the total profit/loss from the bet manager"""
        return self.table.bet_manager.get_total_pnl() if self.table.bet_manager else 0.0

    def get_total_pnl_DKK(self) -> float:
        """Gets the total profit/loss from the bet manager in DKK"""
        return self.table.bet_manager.get_total_pnl() if self.table.bet_manager else 0.0

    def export_bets_to_csv(self):
        """Exports the current session's bets to a CSV file"""
        if self.table.bet_manager:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bets_{timestamp}.csv"
            file_path = f"results/{filename}"
            self.table.bet_manager.export_to_csv(file_path)

    def reset_table(self):
        """Reset table-specific context for a new table"""
        self.table.bet_manager = None
        self.table.line_start_time = None
        self.game.current_mode = ""
        self.game.initial_mode = ""
        self.game.current_bet = None
        self.game.last_bet = None
        self.game.game_result = ""
        self.game.end_line_reason = ""
        self.game.bias = ""
        self.game.outcomes = []  # Reset outcomes when switching tables

    def second_shoe_mode(self):
        """Switch to second shoe mode"""
        self.game.is_second_shoe = True
        self.game.current_mode = self.game.initial_mode
        self.game.current_bet = None
        self.game.last_bet = None
        self.game.game_result = ""
        self.game.end_line_reason = ""
        self.game.first_shoe_outcomes = self.game.outcomes.copy()
        self.game.outcomes = []  # Reset outcomes when switching tables
        self.game.first_shoe_drawdown = self.get_total_pnl()

    def export_line_to_csv(self):
        """Exports the finished line data to a CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = "results/finished_lines.csv"

        # Get first 6 from stored outcomes
        if self.game.is_second_shoe:
            first_6 = get_first_6_non_ties(self.game.first_shoe_outcomes)
        else:
            first_6 = get_first_6_non_ties(self.game.outcomes)
        first_6_str = ''.join(first_6) if first_6 else ''

        if self.game.is_second_shoe:
            # Prepare the line data
            line_data = {
                'timestamp': timestamp,
                'duration': round((datetime.now() - self.table.line_start_time).total_seconds() / 60),
                'initial_mode': self.game.initial_mode,
                'end_line_reason': self.game.end_line_reason,
                'bias': self.game.bias,
                'first_6_outcomes': first_6_str,
                'profit': self.get_total_pnl(),
                'all_outcomes_first_shoe': ''.join(self.game.first_shoe_outcomes),
                'all_outcomes_second_shoe': ''.join(self.game.outcomes)
            }
        else:
            # Prepare the line data
            line_data = {
                'timestamp': timestamp,
                'duration': round((datetime.now() - self.table.line_start_time).total_seconds() / 60),
                'initial_mode': self.game.initial_mode,
                'end_line_reason': self.game.end_line_reason,
                'bias': self.game.bias,
                'first_6_outcomes': first_6_str,
                'profit': self.get_total_pnl(),
                'all_outcomes_first_shoe': ''.join(self.game.outcomes),
                'all_outcomes_second_shoe': ''
            }

        # Create file with headers if it doesn't exist
        file_exists = os.path.exists(file_path)
        headers = line_data.keys()

        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(line_data)

        # Google Sheets export is now handled directly in EndLineState