from dataclasses import dataclass, asdict, field
from typing import Optional
import csv
import threading
import logging
from typing import List
from datetime import datetime

@dataclass
class Bet:
    side: str         # e.g., "P" or "B"
    size: float       # Amount bet
    result: Optional[str] = None  # "W", "L", or "T"
    pnl: float = field(default=0, init=False)  # Profit/loss; calculated after result is set

    def set_result(self, result: str) -> None:
        """
        Sets the result of the bet and calculates the profit or loss.
        
        :param result: A string indicating the outcome. 
                       "W" for win, "L" for loss, "T" for tie.
        """
        self.result = result
        if result == "T":
            self.pnl = 0.0
        elif result == "L":
            self.pnl = -self.size
        else:  # Assuming a win scenario
            # For example, if betting on Banker ("B"), the payout might be 0.95x the size
            self.pnl = self.size * 0.95 if self.side.upper() == "B" else self.size

    def to_dict(self) -> dict:
        """
        Converts the Bet instance to a dictionary.
        
        :return: Dictionary representation of the bet.
        """
        return asdict(self)

class BetManager:
    """Manages bets and calculates PNL"""
    def __init__(self, initial_pnl: float = 0.0):
        self.bets = []
        self._pnl = initial_pnl
        
    def add_bet(self, bet):
        """Add a bet to the list of bets"""
        self.bets.append(bet)
        
    def get_all_bets(self):
        """Return all bets"""
        return self.bets
        
    def get_total_pnl(self):
        """Calculate total PNL from all bets"""
        bet_pnl = sum(bet.pnl for bet in self.bets)
        return round(bet_pnl + self._pnl, 2)  # Include initial PNL from first shoe

    def get_bet_start_time(self) -> datetime:
        """Return the time when the current line started."""
        return self._line_start_time

    def get_number_of_ties(self) -> int:
        """Return the number of ties in the bet history."""
        return sum(1 for bet in self.bets if bet.result == "T")

    def export_to_csv(self, file_path: str):
        """Export all bets to a CSV file."""
        if not self.bets:
            logging.warning("No bets to export.")
            return
        with open(file_path, mode='w', newline='') as file:
            fieldnames = ['side', 'size', 'result', 'pnl']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for bet in self.bets:
                writer.writerow(bet.to_dict())
        logging.info(f"Bets exported to {file_path}")