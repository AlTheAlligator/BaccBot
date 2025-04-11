from core.interaction import click_button, random_mouse_move
from core.screencapture import capture_game_result, get_banker_bet_coordinates, get_player_bet_coordinates, get_10_bet_coordinates, get_50_bet_coordinates, get_250_bet_coordinates, get_500_bet_coordinates, get_1000_bet_coordinates, get_2000_bet_coordinates, get_bet_allowed_coordinates, get_empty_history_coordinates
import time
import random
from core.ocr import get_game_result

def get_game_result_wrapper():
    # Capture the result and pass it to the OCR function
    results = capture_game_result()
    if results is None:
        return "Waiting for Results"
    # results will be either a PIL Image or a file path depending on debug mode
    return get_game_result(results)

def get_bet_allowed():
    coordinates = get_bet_allowed_coordinates()
    if coordinates is None:
        return False
    return True

def shoe_finished():
    coordinates = get_empty_history_coordinates()
    if coordinates is None:
        return False
    return True

def place_bets(side, chips):
    last_chip = None
    for chip in chips:
        if chip != last_chip:
            if chip == 10:
                click_button(get_10_bet_coordinates())
            elif chip == 50:
                click_button(get_50_bet_coordinates())
            elif chip == 250:
                click_button(get_250_bet_coordinates())
            elif chip == 500:
                click_button(get_500_bet_coordinates())
            elif chip == 1000:
                click_button(get_1000_bet_coordinates())
            elif chip == 2000:
                click_button(get_2000_bet_coordinates())

        if side == "P":
            click_button(get_player_bet_coordinates())
        elif side == "B":
            click_button(get_banker_bet_coordinates())

        last_chip = chip

        time.sleep(random.uniform(0.05, 0.1))  # Delay between bets