# Implements the betting strategy, including handling double chops and dynamic mode switching.
from datetime import datetime
import logging
from core.ocr import extract_cubes_and_numbers
from core.screencapture import capture_nameless_cubes
from core.nameless import press_banker_only_btn, press_player_only_btn, is_line_done

CHIP_SIZES = [
    2000,
    1000,
    500,
    250,
    50,
    10
]

def get_first_6_non_ties(outcomes):
    """
    Get the first 6 Player (P) or Banker (B) outcomes, ignoring ties (T).
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'T', 'P', 'B', 'B', 'T']).
    
    Returns:
    - List of the first 6 non-tie outcomes.
    """
    non_tie_outcomes = [outcome for outcome in outcomes if outcome in ['P', 'B']]
    return non_tie_outcomes[:6]

def get_last_6_without_ties(outcomes):
    """
    Get the last 6 Player (P) or Banker (B) outcomes, ignoring ties (T).
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'T', 'P', 'B', 'B', 'T']).
    
    Returns:
    - List of the last 6 non-tie outcomes.
    """
    non_tie_outcomes = [outcome for outcome in outcomes if outcome in ['P', 'B']]
    return non_tie_outcomes[-6:]

def get_outcomes_without_ties(outcomes):
    """
    Get all outcomes without ties (T).
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'T', 'P', 'B', 'B', 'T']).
    
    Returns:
    - List of non-tie outcomes.
    """
    return [outcome for outcome in outcomes if outcome in ['P', 'B']]

def get_outcomes_without_not_played(outcomes):
    """
    Get all outcomes without not played (N).
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'N', 'P', 'B', 'B', 'N']).
    
    Returns:
    - List of non-not played outcomes.
    """
    return [outcome for outcome in outcomes if outcome in ['P', 'B', 'T']]

def analyze_first_6(outcomes, bias, skip_above_games = 10):
    """
    Analyze the first 6 games and determine the starting mode.
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    - skip_above_games: Number of games before too many games have been played and table should be skipped.
    
    Returns:
    - mode: 'PPP', 'BBB', or 'Skip'.
    """
    double_chop_patterns = [
        ['B', 'P', 'P', 'B', 'P', 'P'],
        ['P', 'B', 'B', 'P', 'B', 'B'],
        ['B', 'B', 'P', 'B', 'B', 'P'],
        ['P', 'P', 'B', 'P', 'P', 'B'],
        ['P', 'P', 'B', 'B', 'P', 'P'],
        ['B', 'B', 'P', 'P', 'B', 'B']
    ]

    outcome_played = get_outcomes_without_not_played(outcomes)
    if skip_above_games != 0 and len(outcome_played) > skip_above_games:
        logging.info("Too many games played. Skipping the shoe.")
        return "Skip"
    
    if len(outcome_played) < 6:
        logging.info("Not enough games played. Analyzing...")
        return "Analyzing"
    
    # Check for double chop patterns
    if get_first_6_non_ties(outcomes) in double_chop_patterns:
        logging.info("Double chop detected. Skipping the shoe.")
        return "Skip"  # Skip the shoe
    
    first_6_outcomes = outcome_played[:6]
    tie_count = first_6_outcomes.count('T')
    if tie_count >= 1 and len(outcome_played) > 6:
        first_6_outcomes = outcomes[:7]

    player_count = first_6_outcomes.count('P')
    banker_count = first_6_outcomes.count('B')
    tie_count = first_6_outcomes.count('T')

    

    if tie_count >= 2:
        logging.info("Too many ties. Skipping the shoe.")
        return "Skip"  # Skip the shoe
    if player_count in [4, 5]:
        if bias != "B":
            if has_streaks_threshold_hit(get_first_6_non_ties(outcomes), "PPP", 1):
                return "BBB"
    if banker_count in [4, 5]:
        if bias != "P":
            if has_streaks_threshold_hit(get_first_6_non_ties(outcomes), "BBB", 1):
                return "PPP"
    logging.info("No pattern detected. Skipping...")
    return "Skip"  # Default to skipping if no valid mode

def analyze_first_6_rule(outcomes):
    """
    Analyze the bias of the outcomes.
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    
    Returns:
    - bias: 'P', 'B', or 'T'.
    """
    first_6 = analyze_first_6(outcomes, "", 0)
    if first_6 == "Skip":
        return ("Skip", True)
    elif first_6 == "Analyzing":
        return ("Skip", True)
    
    outcomes_no_ties = get_outcomes_without_ties(outcomes)
    logging.info(f"Analyzing first 6 rule as {first_6} on outcomes: {outcomes_no_ties}")
    if first_6 == "PPP":
        if outcomes_no_ties.count('B') >= outcomes_no_ties.count('P'):
            return ("PPP", True)
        else:
            return ("PPP", False)
    elif first_6 == "BBB":
        if outcomes_no_ties.count('P') >= outcomes_no_ties.count('B'):
            return ("BBB", True)
        else:
            return ("BBB", False)

def analyze_bias(outcomes, minimum_outcomes = 24, bias_threshold = 65):
    """
    Analyze the bias of the outcomes.
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    
    Returns:
    - bias: 'P', 'B', or 'T'.
    """
    outcomes_no_ties = get_outcomes_without_ties(outcomes)
    logging.info(f"Analyzing table: {outcomes_no_ties}")
    if len(outcomes_no_ties) < minimum_outcomes:
        logging.info("Not enough outcomes to analyze.")
        return None
    player_count = outcomes_no_ties.count('P')
    banker_count = outcomes_no_ties.count('B')
    if player_count >= len(outcomes_no_ties) / 100 * bias_threshold:
        logging.info("Player bias detected.")
        return 'P'
    elif banker_count >= len(outcomes_no_ties) / 100 * bias_threshold:
        logging.info("Banker bias detected.")
        return 'B'
    else:
        logging.info("No bias detected.")
        return ""

def play_mode(current_mode, outcomes, inital_mode, last_bet):
    """
    Determine the next bet and handle switching between modes.

    Args:
    - current_mode: 'PPP', 'BBB', or 'Switch'.
    - outcomes: List of all outcomes so far.
    - last_6: Last 6 outcomes for pattern analysis.

    Returns:
    - next_bet: 'P', 'B', or 'T'.
    - updated_mode: Updated mode ('PPP', 'BBB', or 'Switch').
    """
    if current_mode == "Switch":
        # Switch mode: alternate between Player and Banker
        next_bet = 'B' if last_bet.side == 'P' else 'P'
        # Check if switching should stop
        if not find_3_consecutive_losses(outcomes, inital_mode):
            updated_mode = inital_mode
            if inital_mode == "PPP":
                press_banker_only_btn()
                next_bet = 'B'
            else:
                press_player_only_btn()
                next_bet = 'P'
        else:
            updated_mode = "Switch"
    else:
        # PPP or BBB mode
        next_bet = 'B' if current_mode == "PPP" else 'P'
        # Check for 3 consecutive losses
        if current_mode == "PPP" and find_3_consecutive_losses(outcomes, inital_mode):
            updated_mode = "Switch"
            next_bet = "P"
            press_banker_only_btn()
        elif current_mode == "BBB" and find_3_consecutive_losses(outcomes, inital_mode):
            updated_mode = "Switch"
            next_bet = "B"
            press_player_only_btn()
        else:
            updated_mode = current_mode

    return next_bet, updated_mode

def find_3_consecutive_losses(outcomes, bias):
    """
    Find 3 consecutive losses in the last 6 outcomes.
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    
    Returns:
    - True if 3 consecutive losses are found, False otherwise.
    """
    last_6 = get_last_6_without_ties(outcomes)
    in_a_row = 0
    for outcome in last_6:
        if bias == 'BBB':
            if outcome == 'B':
                in_a_row += 1
            else:
                in_a_row = 0
        if bias == 'PPP':
            if outcome == 'P':
                in_a_row += 1
            else:
                in_a_row = 0
        if in_a_row == 3:
            return True
            

def determine_chip_amount(bet_size):
    """
    Determine the chip amount based on the bet size.
    """

    remaining = bet_size
    chips = []
    for chip in CHIP_SIZES:
        while remaining > 0:
            if remaining >= chip:
                chips.append(chip)
                remaining -= chip
            else:
                break

    return chips
    
def cube_threshold_hit(small_cube_threshold = 25, big_cube_threshold = 50, cube_count_threshold = 2, cube_count = 0, extracted_numbers = []):
    """
    Analyze the cubes and extract the numbers.
    """
    if cube_count == 0:
        cube_count, extracted_numbers = extract_cubes_and_numbers(capture_nameless_cubes())

    if cube_count <= cube_count_threshold:
        result = all(x <= small_cube_threshold for x in extracted_numbers)
        if result:
            logging.info(f"{cube_count_threshold} Small cubes detected. Closing the line.")
            return True
        result = all(x >= big_cube_threshold for x in extracted_numbers)
        if result:
            logging.info(f"{cube_count_threshold} Big cubes detected. Closing the line.")
            return True
            
    logging.info(f"Number of cubes: {cube_count}")
    logging.info(f"Extracted numbers: {extracted_numbers}")

def bad_streaks_threshold_hit(context, threshold = 3):
    outcomes_no_ties = get_outcomes_without_ties(context.game.outcomes)

    has_streaks_threshold_hit(outcomes_no_ties, context.game.initial_mode, threshold)
    
def has_streaks_threshold_hit(outcomes, initial_mode, threshold = 1):
    return get_streaks(outcomes, initial_mode) >= threshold
    
def get_streaks(outcomes, mode):
    if mode == "PPP":
        bad_outcome = "P"
    else:
        bad_outcome = "B"
    streaks = 0
    in_a_row = 0
    for outcome in outcomes:
        if outcome == bad_outcome:
            in_a_row += 1
            if in_a_row == 3:
                streaks += 1
        else:
            in_a_row = 0
        
    return streaks
def check_for_end_line(context, streak_threshold = 3, min_profit = -50, use_mini_line_exit = False, use_moderate_exit = False):
    if is_line_done():
        context.game.end_line_reason = "nameless built-in line end"
        return True
    
    current_pnl = context.get_total_pnl()
    if current_pnl <= -4000:
        context.game.end_line_reason = "max drawdown reached (-4000)"
        return True
    
    cube_count, extracted_numbers = extract_cubes_and_numbers(capture_nameless_cubes())
    if current_pnl >= min_profit and bad_streaks_threshold_hit(context, streak_threshold):
        if cube_threshold_hit(50, 50, 3, cube_count=cube_count, extracted_numbers=extracted_numbers):
            context.game.end_line_reason = "3+ bad streak, at least -50 profit and less than 3 cubes"
            return True
    if cube_threshold_hit(10, 100, 1, cube_count=cube_count, extracted_numbers=extracted_numbers):
        context.game.end_line_reason = "only 1 cube value left, lower than 10 or higher than 100 detected"
        return True
    
    if use_mini_line_exit:
        if check_for_mini_line_exit(context, cube_count, extracted_numbers):
            return True
    
    if use_moderate_exit:
        if check_for_moderate_exit(context, cube_count, extracted_numbers):
            return True
    
    return False
    
def check_for_mini_line_exit(context, cube_count, extracted_numbers):
    start_time = context.table.line_start_time
    duration_minutes = (datetime.now() - start_time).total_seconds() / 60
    pnl = context.get_total_pnl()
    outcomes_no_ties = get_outcomes_without_ties(context.game.outcomes)
    streaks = get_streaks(outcomes_no_ties, context.game.initial_mode)

    # If 200+ profit within 5 minutes, exit
    if duration_minutes <= 5:
        if pnl >= 200:
            context.game.end_line_reason = "quick profit (200+ in 5 min)"
            return True
    # If 250+ profit within 10 minutes, exit. If 150+ profit within 10 minutes and 2 bad streaks, exit.
    elif duration_minutes <= 10:
        if pnl >= 250:
            context.game.end_line_reason = "good profit (250+ in 10 min)"
            return True
        if pnl >= 150:
            if streaks >= 2:
                context.game.end_line_reason = "decent profit with bad streaks (150+ in 10 min, 2 streaks)"
                return True
    # If 300+ profit within 15 minutes, exit.
    elif duration_minutes > 10:
        if pnl >= 300:
            context.game.end_line_reason = "excellent profit (300+)"
            return True
        
    # If 150+ profit and 3 bad streaks, exit.
    if streaks >= 3:
        if pnl >= 150:
            context.game.end_line_reason = "decent profit with many bad streaks (150+, 3 streaks)"
            return True
        
    # If 4 bad streaks and profit, exit.
    if streaks >= 4:
        if pnl >= 0:
            context.game.end_line_reason = "profitable with too many bad streaks (4 streaks)"
            return True
        
    bets = context.table.bet_manager.get_all_bets()
    ties = context.table.bet_manager.get_number_of_ties()
    # If less than 6 bets and 2 ties, exit.
    if len(bets) <= 6 and ties >= 2 and pnl > 0:
        context.game.end_line_reason = "too many early ties (<6 bets, 2 ties)"
        return True
    
    # If less than 12 bets and 3 ties, exit.
    if len(bets) <= 12 and ties >= 3 and pnl > 0:
        context.game.end_line_reason = "too many early ties (<12 bets, 3 ties)"
        return True
    
    # If less than 4 cubes and above 100 size, exit.
    if cube_threshold_hit(1, 100, 3, cube_count=cube_count, extracted_numbers=extracted_numbers):
        context.game.end_line_reason = "large cube values with few cubes"
        return True
    
    return False

def check_for_moderate_exit(context):
    pnl = context.get_total_pnl()
    outcomes_no_ties = get_outcomes_without_ties(context.game.outcomes)
    streaks = get_streaks(outcomes_no_ties, context.game.initial_mode)

    # If 150+ profit and 3 bad streaks, exit.
    if streaks >= 3:
        if pnl >= 150:
            context.game.end_line_reason = "decent profit with many bad streaks (150+, 3 streaks)"
            return True