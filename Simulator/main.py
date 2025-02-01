from modules.screen_capture import capture_full_screen, get_game_window_coordinates, capture_game_window, capture_history
from modules.template_matching import crop_history_box, template_match, result_templates
from modules.interaction import interact_with_game
from modules.utils import log, start_keyboard_listener, stop_event,logging
from modules.strategy import analyze_first_6, play_mode

def main():
    logging.info("Starting Baccarat automation...")

    # Start the keyboard listener
    start_keyboard_listener()

    # Locate game window
    game_window_coordinates = get_game_window_coordinates("Screenshots/full_screen.png", "Screenshots/GameScreen.png")
    if not game_window_coordinates:
        logging.error("Game window not found. Exiting.")
        return

    game_state = "init"
    next_bet = ""

    # Main interaction loop
    while not stop_event.is_set():
        capture_game_window(game_window_coordinates)
        cropped_image = capture_history(game_window_coordinates)
        outcomes = template_match(cropped_image, result_templates)
        if game_state in ["init", "analyzing"]:
            game_state = analyze_first_6(outcomes)
            log(f"Detected outcomes: {outcomes[:7]}")
        else:
            if game_state == "betting":
                next_bet, game_state = play_mode(game_state, outcomes)
            interact_with_game(game_window_coordinates, buttons, game_state, next_bet)

    logging.info("Program terminated by user.")
        
        

if __name__ == "__main__":
    main()
