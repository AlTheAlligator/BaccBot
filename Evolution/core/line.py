"""
Line object for nameless betting system.

This module implements the nameless betting system where bet sizes are determined
by adding the first cube from each of two lists (left_cubes and right_cubes).
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class Bet:
    """
    Represents a single bet in the nameless betting system.
    """
    VALID_RESULTS = ['W', 'L', 'T']  # Win, Loss, Tie

    def __init__(self, side: str, size: float):
        """
        Initialize a bet.

        Args:
            side: 'P' for Player or 'B' for Banker
            size: Size of the bet
        """
        if side not in ['P', 'B']:
            raise ValueError("side must be 'P' or 'B'")

        self.side = side
        self.size = size
        self.result = None
        self.profit = 0.0
        self.commission = 0.0

    def set_result(self, result: str) -> float:
        """
        Set the result of the bet and calculate profit/loss.

        Args:
            result: 'W' for win, 'L' for loss, 'T' for tie

        Returns:
            float: The profit/loss from this bet

        Raises:
            ValueError: If result is not valid
            RuntimeError: If result is set multiple times
        """
        if result not in self.VALID_RESULTS:
            raise ValueError(f"Invalid result: {result}. Must be one of: {self.VALID_RESULTS}")

        if self.result is not None:
            raise RuntimeError("Cannot set result multiple times")

        self.result = result

        if result == 'W':
            # Account for commission on banker bets
            if self.side == 'B':
                self.commission = self.size * 0.05
                self.profit = self.size - self.commission
            else:
                self.profit = self.size
        elif result == 'L':
            self.profit = -self.size
        else:  # Tie
            self.profit = 0

        return self.profit

    def __str__(self) -> str:
        """String representation of the bet."""
        result_str = self.result if self.result else 'Pending'
        return f"Bet({self.side}, size={self.size}, result={result_str}, profit={self.profit})"


class Line:
    """
    Represents a betting line using the nameless betting system.

    The nameless system uses two lists of cubes (left_cubes and right_cubes) to determine
    bet sizes. The bet size is the sum of the first cube from each list. After a win,
    the first cube from each list is removed. After a loss, the current bet size is added
    as a cube to the right_cubes. If right_cubes has 2 or more cubes after adding,
    the last cube is moved to the end of left_cubes.
    """

    def __init__(self, initial_side: str, left_cubes: List[int] = None, right_cubes: List[int] = None, stop_loss: float = -4000):
        """
        Initialize a betting line.

        Args:
            initial_side: Initial betting side ('P' for Player or 'B' for Banker)
            left_cubes: Initial left cubes (default: [1])
            right_cubes: Initial right cubes (default: [1])
            stop_loss: PnL threshold at which to stop the line (default: -4000)
        """
        if initial_side not in ['P', 'B']:
            raise ValueError("initial_side must be 'P' or 'B'")

        self.initial_side = initial_side
        self.current_side = initial_side
        self.left_cubes = left_cubes if left_cubes is not None else [4,5,6,7,8,9,10,11,12,13,14]
        self.right_cubes = right_cubes if right_cubes is not None else [44,40,36,33,30,27,24,21,18,17,16,15]
        self.bets: List[Bet] = []
        self.pnl = 0.0
        self.commission_paid = 0.0
        self.is_active = True
        self.outcomes: List[str] = []
        self.stop_loss = stop_loss

    def get_current_bet_size(self) -> Optional[int]:
        """
        Calculate the current bet size based on the first cube from each list.

        Returns:
            int: Current bet size or None if either list is empty
        """
        bet_size = 0

        if self.left_cubes:
            bet_size += self.left_cubes[0]
        if self.right_cubes:
            bet_size += self.right_cubes[0]

        return bet_size

    def place_bet(self, side: str) -> Optional[Bet]:
        """
        Place a bet with the current bet size.

        Args:
            side: Betting side ('P' for Player or 'B' for Banker)

        Returns:
            Bet: The placed bet or None if the line is not active
        """
        if not self.is_active:
            logger.info("Line is not active, cannot place bet")
            return None

        bet_size = self.get_current_bet_size()
        if bet_size == 0:
            logger.info("Cannot place bet: Line is finished")
            self.is_active = False
            return None

        bet = Bet(side, bet_size)
        self.bets.append(bet)
        self.current_side = side
        return bet

    def process_outcome(self, outcome: str) -> Dict[str, Any]:
        """
        Process a game outcome and update the line state.

        Args:
            outcome: Game outcome ('P', 'B', or 'T')

        Returns:
            dict: Result information including profit, updated cube lists, etc.
        """
        self.outcomes.append(outcome)

        if not self.is_active or not self.bets:
            return {
                "is_active": self.is_active,
                "left_cubes": self.left_cubes.copy(),
                "right_cubes": self.right_cubes.copy(),
                "profit": 0,
                "total_pnl": self.pnl,
                "commission_paid": self.commission_paid
            }

        # Get the last bet
        last_bet = self.bets[-1]

        # Determine result
        if outcome == 'T':
            result = 'T'  # Tie
        elif outcome == last_bet.side:
            result = 'W'  # Win
        else:
            result = 'L'  # Loss

        # Set the result and update PnL
        profit = last_bet.set_result(result)
        self.pnl += profit

        if last_bet.side == 'B' and result == 'W':
            self.commission_paid += last_bet.commission

        # Check if PnL has gone below the stop-loss threshold
        if self.pnl < self.stop_loss:
            self.is_active = False
            logger.warning(f"Line stopped: PnL ({self.pnl:.2f}) below stop-loss threshold of {self.stop_loss}")

        # Update cube lists based on result
        if result == 'W':
            # Remove first cube from each list after a win
            if self.left_cubes:
                self.left_cubes.pop(0)
            if self.right_cubes:
                self.right_cubes.pop(0)
        elif result == 'L':
            # Add current bet size as a cube to right_cubes after a loss
            bet_size = last_bet.size
            self.right_cubes.insert(0, bet_size)  # Insert at the beginning

            # If right_cubes has 2 or more cubes than left_cubes, move the last cube to left_cubes
            if len(self.right_cubes) > len(self.left_cubes) + 1:
                last_cube = self.right_cubes.pop()
                self.left_cubes.append(last_cube)

        # Check if the line is still active
        if not self.left_cubes and not self.right_cubes:
            self.is_active = False
            logger.info("Line is no longer active: one or both cube lists are empty")

        return {
            "is_active": self.is_active,
            "left_cubes": self.left_cubes.copy(),
            "right_cubes": self.right_cubes.copy(),
            "profit": profit,
            "total_pnl": self.pnl,
            "commission_paid": self.commission_paid
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the line.

        Returns:
            dict: Current line state
        """
        return {
            "initial_side": self.initial_side,
            "current_side": self.current_side,
            "left_cubes": self.left_cubes.copy(),
            "right_cubes": self.right_cubes.copy(),
            "is_active": self.is_active,
            "pnl": self.pnl,
            "commission_paid": self.commission_paid,
            "bet_count": len(self.bets),
            "outcome_count": len(self.outcomes),
            "current_bet_size": self.get_current_bet_size()
        }

    def __str__(self) -> str:
        """String representation of the line."""
        return (f"Line(side={self.current_side}, active={self.is_active}, "
                f"left_cubes={self.left_cubes}, right_cubes={self.right_cubes}, "
                f"pnl={self.pnl:.2f}, bets={len(self.bets)})")


class LineManager:
    """
    Manages multiple betting lines and provides aggregated statistics.
    """

    def __init__(self):
        """Initialize the line manager."""
        self.lines: List[Line] = []
        self.active_line: Optional[Line] = None
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.tie_count = 0

    def create_line(self, initial_side: str, left_cubes: List[int] = None, right_cubes: List[int] = None, stop_loss: float = -4000) -> Line:
        """
        Create a new betting line and set it as the active line.

        Args:
            initial_side: Initial betting side ('P' for Player or 'B' for Banker)
            left_cubes: Initial left cubes (default: [1])
            right_cubes: Initial right cubes (default: [1])
            stop_loss: PnL threshold at which to stop the line (default: -4000)

        Returns:
            Line: The newly created line
        """
        line = Line(initial_side, left_cubes, right_cubes, stop_loss)
        self.lines.append(line)
        self.active_line = line
        return line

    def place_bet(self, side: str) -> Optional[Bet]:
        """
        Place a bet on the active line.

        Args:
            side: Betting side ('P' for Player or 'B' for Banker)

        Returns:
            Bet: The placed bet or None if no active line
        """
        if not self.active_line or not self.active_line.is_active:
            logger.warning("No active line available for betting")
            return None

        return self.active_line.place_bet(side)

    def process_outcome(self, outcome: str) -> Dict[str, Any]:
        """
        Process a game outcome for the active line.

        Args:
            outcome: Game outcome ('P', 'B', or 'T')

        Returns:
            dict: Result information
        """
        if not self.active_line:
            logger.warning("No active line to process outcome")
            return {"error": "No active line"}

        result = self.active_line.process_outcome(outcome)

        # Update overall statistics
        self.total_pnl = sum(line.pnl for line in self.lines)
        self.total_commission = sum(line.commission_paid for line in self.lines)

        # Update win/loss/tie counts
        if self.active_line.bets and self.active_line.bets[-1].result:
            last_result = self.active_line.bets[-1].result
            if last_result == 'W':
                self.win_count += 1
            elif last_result == 'L':
                self.loss_count += 1
            elif last_result == 'T':
                self.tie_count += 1

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for all lines.

        Returns:
            dict: Aggregated statistics
        """
        total_bets = self.win_count + self.loss_count + self.tie_count
        win_rate = self.win_count / total_bets if total_bets > 0 else 0

        return {
            "total_lines": len(self.lines),
            "active_lines": sum(1 for line in self.lines if line.is_active),
            "total_pnl": self.total_pnl,
            "total_commission": self.total_commission,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "total_bets": total_bets,
            "win_rate": win_rate
        }

    def __str__(self) -> str:
        """String representation of the line manager."""
        active_count = sum(1 for line in self.lines if line.is_active)
        return (f"LineManager(lines={len(self.lines)}, active={active_count}, "
                f"pnl={self.total_pnl:.2f}, win_rate={self.win_count/(self.win_count+self.loss_count)*100:.1f}% if self.win_count+self.loss_count > 0 else 0)")


def simulate_line(outcomes: List[str], initial_side: str,
                 left_cubes: List[int] = None, right_cubes: List[int] = None,
                 strategy_func = None, stop_loss: float = -4000) -> Tuple[Line, Dict[str, Any]]:
    """
    Simulate a betting line with the given outcomes.

    Args:
        outcomes: List of game outcomes ('P', 'B', 'T')
        initial_side: Initial betting side ('P' for Player or 'B' for Banker)
        left_cubes: Initial left cubes (default: [1])
        right_cubes: Initial right cubes (default: [1])
        strategy_func: Optional function to determine betting side (default: always bet initial_side)
        stop_loss: PnL threshold at which to stop the line (default: -4000)

    Returns:
        tuple: (Line object, Statistics dictionary)
    """
    line = Line(initial_side, left_cubes, right_cubes, stop_loss)

    # Default strategy: always bet the initial side
    if strategy_func is None:
        strategy_func = lambda outcomes, line: initial_side

    statistics = {
        "total_bets": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "final_pnl": 0,
        "max_pnl": 0,
        "min_pnl": 0,
        "max_drawdown": 0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "current_consecutive_wins": 0,
        "current_consecutive_losses": 0
    }

    # Process each outcome
    for outcome in outcomes:
        # Skip if line is no longer active
        if not line.is_active:
            break

        # Determine betting side using strategy function
        betting_side = strategy_func(line.outcomes, line)

        # Place bet
        bet = line.place_bet(betting_side)
        if bet is None:
            break

        # Process outcome
        result = line.process_outcome(outcome)

        # Update statistics
        statistics["total_bets"] += 1
        statistics["final_pnl"] = line.pnl
        statistics["max_pnl"] = max(statistics["max_pnl"], line.pnl)
        statistics["min_pnl"] = min(statistics["min_pnl"], line.pnl)

        # Calculate drawdown
        current_drawdown = statistics["max_pnl"] - line.pnl
        statistics["max_drawdown"] = max(statistics["max_drawdown"], current_drawdown)

        # Update win/loss statistics
        if bet.result == 'W':
            statistics["wins"] += 1
            statistics["current_consecutive_wins"] += 1
            statistics["current_consecutive_losses"] = 0
            statistics["max_consecutive_wins"] = max(
                statistics["max_consecutive_wins"],
                statistics["current_consecutive_wins"]
            )
        elif bet.result == 'L':
            statistics["losses"] += 1
            statistics["current_consecutive_losses"] += 1
            statistics["current_consecutive_wins"] = 0
            statistics["max_consecutive_losses"] = max(
                statistics["max_consecutive_losses"],
                statistics["current_consecutive_losses"]
            )
        elif bet.result == 'T':
            statistics["ties"] += 1
            # Ties don't affect consecutive win/loss counts

    # Calculate final statistics
    total_decisions = statistics["wins"] + statistics["losses"]
    statistics["win_rate"] = statistics["wins"] / total_decisions if total_decisions > 0 else 0
    statistics["final_left_cubes"] = line.left_cubes.copy()
    statistics["final_right_cubes"] = line.right_cubes.copy()
    statistics["is_completed"] = not line.is_active

    return line, statistics
