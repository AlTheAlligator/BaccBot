import pyautogui
import cv2
from modules.utils import log
from modules.template_matching import locate_button, crop_history_box, template_match, result_templates
from modules.interaction import click_button
from PIL import Image
import time

def capture_full_screen(save_path="Screenshots/full_screen.png"):
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return save_path

def get_game_window_coordinates(screen_path, reference_path, threshold=0.8):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread(reference_path)

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
        print(f"Reference image found at: {x}, {y}")
        # Define the game window relative to the reference image
        game_window = (x, y, 800, 600)  # Adjust width and height based on your game
        return game_window
    log("Game window not found. Exiting.", level="error")
    return None

def capture_game_window(game_window_coordinates, save_path="Screenshots/game_window.png"):
    screenshot = pyautogui.screenshot(region=game_window_coordinates)
    screenshot.save(save_path)
    return save_path

def capture_history(game_window_coordinates, threshold=0.8):
    game_window = capture_game_window(game_window_coordinates)
    location = locate_button(game_window, '/Screenshots/history.png')
    if location:
        game_window_top_left = (game_window_coordinates[0], game_window_coordinates[1])
        click_button(location, game_window_top_left)
        time.sleep(1)
        capture_game_window(game_window_coordinates)
        cropped_image = crop_history_box("Screenshots/game_window.png", (22, 22, 375, 175))
        outcomes = template_match(cropped_image, result_templates)
        log(f"Detected outcomes: {outcomes[:7]}")
        return outcomes