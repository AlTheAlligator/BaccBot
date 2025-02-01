import pyautogui
import time
import random
from modules.template_matching import locate_button

buttons = {
    "Repeat": "Screenshots/repeat.png",
    "Deal": "Screenshots/deal.png",
    "Draw": "Screenshots/draw.png",
}
bet_buttons = {
    "Banker_Bet": "Screenshots/banker_bet.png",
    "Player_Bet": "Screenshots/player_bet.png"
}

def click_button(button_location, game_window_top_left, button_size=(50, 20)):
    adjusted_x = button_location[0] + game_window_top_left[0]
    adjusted_y = button_location[1] + game_window_top_left[1]

    button_center = (adjusted_x + button_size[0] // 2, adjusted_y + button_size[1] // 2)
    pyautogui.moveTo(button_center[0], button_center[1], duration=random.uniform(0.2, 0.5))
    pyautogui.click()

def interact_with_game(game_window_coordinates, buttons, game_state):
    game_window_top_left = (game_window_coordinates[0], game_window_coordinates[1])

    for action, template in buttons.items():
        if game_state == "Skip" and action in ["Banker_Bet", "Player_Bet"]:
            continue
        location = locate_button("Screenshots/game_window.png", template)
        if location:
            click_button(location, game_window_top_left)
            time.sleep(random.uniform(0.5, 1.5))
            return

def place_bet(next_bet, game_window_top_left, button_positions):
    """
    Place a bet by clicking the appropriate button.
    
    Args:
    - next_bet: 'B', 'P', or 'T' (Banker, Player, or Tie).
    - game_window_top_left: Top-left coordinates of the game window.
    - button_positions: Dictionary with button positions relative to the game window.
        Example: {'B': (100, 200), 'P': (200, 200), 'T': (150, 250)}
    """
    if next_bet not in button_positions:
        print(f"Invalid bet: {next_bet}")
        return

    # Get the button's relative position and adjust for the game window
    button_relative = button_positions[next_bet]
    button_absolute = (game_window_top_left[0] + button_relative[0],
                       game_window_top_left[1] + button_relative[1])

    # Simulate the mouse click
    pyautogui.moveTo(button_absolute[0], button_absolute[1], duration=random.uniform(0.2, 0.5))
    pyautogui.click()
    print(f"Placed bet on: {next_bet}")
