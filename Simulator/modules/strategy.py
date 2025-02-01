def analyze_first_6(outcomes):
    """
    Analyze the first 6 games and determine the starting mode.
    
    Args:
    - outcomes: List of first 6 outcomes (e.g., ['P', 'B', 'P', 'B', 'B', 'P']).
    
    Returns:
    - mode: 'PPP', 'BBB', or 'Skip'.
    - start_game: Game number to start betting (7 or 8).
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
    if get_first_6_without_ties(outcomes) in double_chop_patterns:
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
        return "PPP"
    if banker_count in [4, 5]:
        return "BBB"
    return "Skip"  # Default to skipping if no valid mode

def get_first_6_without_ties(outcomes):
    """
    Get the last 6 Player (P) or Banker (B) outcomes, ignoring ties (T).
    
    Args:
    - outcomes: List of outcomes (e.g., ['P', 'B', 'T', 'P', 'B', 'B', 'T']).
    
    Returns:
    - List of the last 6 non-tie outcomes.
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


def play_mode(current_mode, outcomes):
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
    last_6 = get_last_6_without_ties(outcomes)
    if current_mode == "Switch":
        # Switch mode: alternate between Player and Banker
        next_bet = 'B' if outcomes[-1] == 'P' else 'P'
        # Check if switching should stop
        if last_6.count('PPP') == 0 and last_6.count('BBB') == 0:
            updated_mode = analyze_first_6(outcomes)
        else:
            updated_mode = "Switch"
    else:
        # PPP or BBB mode
        next_bet = 'B' if current_mode == "PPP" else 'P'
        # Check for 3 consecutive losses
        if current_mode == "PPP" and last_6[-3:] == ['P', 'P', 'P']:
            updated_mode = "Switch"
        elif current_mode == "BBB" and last_6[-3:] == ['B', 'B', 'B']:
            updated_mode = "Switch"
        else:
            updated_mode = current_mode

    return next_bet, updated_mode
