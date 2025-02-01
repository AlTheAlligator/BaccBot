# Implements the betting strategy, including handling double chops and dynamic mode switching.
import logging
from core.ocr import extract_cubes_and_numbers
from core.screencapture import capture_nameless_cubes, capture_history
from core.nameless import press_banker_only_btn, press_player_only_btn

CHIP_SIZES = {
    2000,
    1000,
    500,
    250,
    50,
    10
}

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

def analyze_first_6(outcomes):
    """
    Analyze the first 6 games and determine the starting mode.
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    
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
    
    # Check for double chop patterns
    if get_first_6_non_ties(outcomes) in double_chop_patterns:
        logging.info("Double chop detected. Skipping the shoe.")
        return "Skip"  # Skip the shoe
    
    first_6_outcomes = outcomes[:6]
    tie_count = first_6_outcomes.count('T')
    if tie_count >= 1:
        first_6_outcomes = outcomes[:7]

    player_count = first_6_outcomes.count('P')
    banker_count = first_6_outcomes.count('B')
    tie_count = first_6_outcomes.count('T')
    not_played_count = first_6_outcomes.count('N')

    if not_played_count > 0:
        return "Analyzing"
    if tie_count >= 2:
        return "Skip"  # Skip the shoe
    if player_count in [4, 5]:
        return "BBB"
    if banker_count in [4, 5]:
        return "PPP"
    return "Skip"  # Default to skipping if no valid mode

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
        next_bet = 'B' if last_bet == 'P' else 'P'
        # Check if switching should stop
        if not find_3_consecutive_losses(outcomes, inital_mode):
            updated_mode = inital_mode
            if inital_mode == "PPP":
                press_banker_only_btn()
            else:
                press_player_only_btn()
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
    
def cube_threshold_hit(small_cube_threshold = 25, big_cube_threshold = 50, cube_count_threshold = 2):
    """
    Analyze the cubes and extract the numbers.
    """
    cube_count, extracted_numbers = extract_cubes_and_numbers(capture_nameless_cubes())

    if cube_count <= cube_count_threshold:
        result = all(x <= small_cube_threshold for x in extracted_numbers)
        if result:
            logging.info("3 Small cubes detected. Closing the line.")
            return True
        result = all(x >= big_cube_threshold for x in extracted_numbers)
        if result:
            logging.info("3 Big cubes detected. Closing the line.")
            return True
            
    logging.info(f"Number of cubes: {cube_count}")
    logging.info(f"Extracted numbers: {extracted_numbers}")

def bad_streaks_threshold_hit(initial_mode, threshold = 3):
    outcomes = get_outcomes_without_ties(capture_history(True))

    if initial_mode == "PPP":
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
        
    if streaks >= threshold:
        return "Close_Positive"
    
def check_for_end_line(initial_mode, current_drawdown, streak_threshold = 3, max_drawdown = 300):
    end_line = False
    if bad_streaks_threshold_hit(initial_mode, streak_threshold):
        end_line = True
    if cube_threshold_hit():
        end_line = True

    if end_line and max_drawdown >= current_drawdown:
        return True
    
    return False
    