# Handles detecting game states like "Before Betting," "In Progress," and "Waiting for Results."
import cv2
#import yaml

# Load configuration
#with open("assets/config.yaml", "r") as config_file:
#    config = yaml.safe_load(config_file)

def detect_game_state(game_window_image, state_templates):
    """
    Detect the current game state by analyzing the game window.

    Args:
    - game_window_image: Path to the screenshot of the game window.
    - state_templates: Dictionary of templates for detecting game states.

    Returns:
    - game_state: The detected game state (e.g., 'Before Betting', 'In Progress').
    """
    game_window = cv2.imread(game_window_image)

    for state, template_path in state_templates.items():
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        result = cv2.matchTemplate(game_window, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > config["thresholds"]["template_matching"]:  # Threshold for matching
            return state

    return "Unknown"