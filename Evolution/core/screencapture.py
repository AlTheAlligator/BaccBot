import pyautogui
import cv2
from PIL import Image
import time
from core.analysis import table_history_template_match, table_match_on_color
import numpy as np
import logging
import os
from pathlib import Path

game_window_coodinates = (None, None, None, None)
home_button_coordinates = (None, None, None, None)
games_button_coordinates = (None, None, None, None)
baccarat_button_coordinates = (None, None, None, None)
nameless_window_coordinates = (None, None, None, None)
nameless_bet_coordinates = (None, None, None, None)
nameless_cube_coordinates = (None, None, None, None)
game_result_coordinates = (None, None, None, None)
bet_allowed_coordinates = (None, None, None, None)
bet_10_coordinates = (None, None, None, None)
bet_50_coordinates = (None, None, None, None)
bet_250_coordinates = (None, None, None, None)
bet_500_coordinates = (None, None, None, None)
bet_1000_coordinates = (None, None, None, None)
bet_2000_coordinates = (None, None, None, None)
bet_5000_coordinates = None
banker_bet_coordinates = (None, None, None, None)
player_bet_coordinates = (None, None, None, None)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def get_game_window_coordinates(threshold=0.5):
    global game_window_coodinates
    if game_window_coodinates[0] is not None:
        return game_window_coodinates

    screen = cv2.imread(capture_full_screen())
    lc_reference = cv2.imread("./assets/templates/corner_left.png")
    rc_reference = cv2.imread("./assets/templates/corner_right.png")

    lc_result = cv2.matchTemplate(screen, lc_reference, cv2.TM_CCOEFF_NORMED)
    rc_result = cv2.matchTemplate(screen, rc_reference, cv2.TM_CCOEFF_NORMED)
    
    _, lc_max_val, _, lc_max_loc = cv2.minMaxLoc(lc_result)
    _, rc_max_val, _, rc_max_loc = cv2.minMaxLoc(rc_result)

    if lc_max_val >= threshold and rc_max_val >= threshold:
        lc_x, lc_y = lc_max_loc
        rc_x, rc_y = rc_max_loc

        # Calculate initial width and height
        raw_width = rc_x - lc_x
        raw_height = rc_y - lc_y

        # Adaptive offsets based on detected width and height
        lc_x_offset = int(raw_width * 0.002)  # Adjust based on a percentage of width
        lc_y_offset = int(raw_height * 0.007)
        rc_x_offset = int(raw_width * 0.07)
        rc_y_offset = int(raw_height * 0.053)

        # Apply offsets to corners
        lc_x -= lc_x_offset
        lc_y -= lc_y_offset
        rc_x += rc_x_offset
        rc_y += rc_y_offset

        # Recalculate width and height
        adjusted_width = rc_x - lc_x
        adjusted_height = rc_y - lc_y

        game_window_coodinates = (lc_x, lc_y, adjusted_width, adjusted_height)
        # Return the adjusted game window's coordinates and size
        return (lc_x, lc_y, adjusted_width, adjusted_height)
    else:
        print("Could not locate game window with sufficient confidence.")
        return None

def get_home_button_coordinates(threshold=0.8):
    global home_button_coordinates
    if home_button_coordinates[0] is not None:
        return home_button_coordinates
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/evolution_home.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        home_button_coordinates = (x, y, 50, 50)
        return home_button_coordinates
    else:
        print("Could not locate home button with sufficient confidence.")

def get_games_button_coordinates():
    global games_button_coordinates
    if games_button_coordinates[0] is not None:
        return games_button_coordinates
    
    home_button_coordinates = get_home_button_coordinates()
    games_button_coordinates = (300, home_button_coordinates[1] + 70, 50, 30)
    return games_button_coordinates

def get_baccarat_button_coordinates(threshold=0.65):
    global baccarat_button_coordinates
    if baccarat_button_coordinates[0] is not None:
        return baccarat_button_coordinates
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/baccarat_btn.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        baccarat_button_coordinates = (x+5, y+5, 50, 50)
        return baccarat_button_coordinates
    else:
        logging.info("Could not locate baccarat button with sufficient confidence.")
        logging.info(f"Max value: {max_val}")

def get_nameless_window_coordinates(threshold=0.8):
    global nameless_window_coordinates
    if nameless_window_coordinates[0] is not None:
        return nameless_window_coordinates
    
    screen = cv2.imread(capture_full_screen())
    lc_reference = cv2.imread("./assets/templates/corner_nameless_left.png")
    rc_reference = cv2.imread("./assets/templates/corner_nameless_right.png")

    lc_result = cv2.matchTemplate(screen, lc_reference, cv2.TM_CCOEFF_NORMED)
    rc_result = cv2.matchTemplate(screen, rc_reference, cv2.TM_CCOEFF_NORMED)
    
    _, lc_max_val, _, lc_max_loc = cv2.minMaxLoc(lc_result)
    _, rc_max_val, _, rc_max_loc = cv2.minMaxLoc(rc_result)

    if lc_max_val >= threshold and rc_max_val >= threshold:
        lc_x, lc_y = lc_max_loc
        rc_x, rc_y = rc_max_loc

        # Calculate initial width and height
        raw_width = rc_x - lc_x

        # Adaptive offsets based on detected width and height
        lc_x_offset = int(raw_width * 0.002)  # Adjust based on a percentage of width
        rc_x_offset = int(raw_width * 0.07)

        # Apply offsets to corners
        lc_x -= lc_x_offset
        rc_x += rc_x_offset

        # Recalculate width and height
        adjusted_width = rc_x - lc_x
        adjusted_height = 1200

        nameless_window_coordinates = (lc_x, lc_y, adjusted_width, adjusted_height)
        # Return the adjusted game window's coordinates and size
        return (lc_x, lc_y, adjusted_width, adjusted_height)
    else:
        print("Could not locate nameless window with sufficient confidence.")
        return None

def get_nameless_bet_coordinates(threshold=0.8):
    global nameless_bet_coordinates
    if nameless_bet_coordinates[0] is not None:
        return nameless_bet_coordinates
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/betbox.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        # Recalculate width and height
        adjusted_x = x - 30
        adjusted_y = y - 15
        adjusted_width = 125
        adjusted_height = 50

        nameless_bet_coordinates = (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
        # Return the adjusted game window's coordinates and size
        return (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
    else:
        print("Could not locate bet box with sufficient confidence.")
        return None
    
def get_nameless_cube_coordinates(threshold=0.8):
    global nameless_cube_coordinates
    if nameless_cube_coordinates[0] is not None:
        return nameless_cube_coordinates
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/cubeidentify.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
    else:
        reference = cv2.imread("./assets/templates/cubeidentify_init.png")

        result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            x, y = max_loc
        else:
            print("Could not locate cubes with sufficient confidence.")
            return None
    # Recalculate width and height
    adjusted_x = x - 125
    adjusted_y = y + 40
    adjusted_width = 450
    adjusted_height = 200

    nameless_cube_coordinates = (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
    # Return the adjusted game window's coordinates and size
    return (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
    
def get_game_result_coordinates(threshold=0.6):
    coordinates = get_game_window_coordinates()
    # Recalculate width and height
    adjusted_x = (coordinates[0] + coordinates[2] // 2) - 50
    adjusted_y = (coordinates[1] + coordinates[3] // 2) - 310
    adjusted_width = 120
    adjusted_height = 50

    # Return the adjusted game window's coordinates and size
    return (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
    
def get_bet_allowed_coordinates(threshold=0.8, first_run=True):
    global bet_allowed_coordinates
    
    reference = cv2.imread("./assets/templates/bet_allowed.png")
    #if bet_allowed_coordinates[0] is None:
    screen = cv2.imread(capture_full_screen())
    scales = np.linspace(0.9, 1.1, 3)
    #else:
    #    img = Image.open(capture_coordinates(bet_allowed_coordinates))
    #    if table_match_on_color(np.array(img), {"Allowed": (106, 58, 141), "Not Allowed": (29, 26, 23)}) == "Allowed":
    #        return bet_allowed_coordinates
    #    else:
    #       return None
    
    best_val = 0
    best_loc = None
    for scale in scales:
        resized_reference = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if resized_reference.shape[0] > screen.shape[0] or resized_reference.shape[1] > screen.shape[1]:
            scale_factor = min(screen.shape[0] / resized_reference.shape[0], screen.shape[1] / resized_reference.shape[1])
            resized_reference = cv2.resize(resized_reference, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        result = cv2.matchTemplate(screen, resized_reference, cv2.TM_CCOEFF_NORMED)
    
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val and max_val >= threshold:
            best_val = max_val
            best_loc = max_loc
    
    if best_loc == None:
        print("Could not locate bet allowed with sufficient confidence.")
        return None
    
    if first_run:
        get_bet_allowed_coordinates(threshold=0.8, first_run=False)

    x, y = best_loc
    screenshot = pyautogui.screenshot(region=(x-10, y-10, 40, 40))
    screenshot.save("./assets/screenshots/test.png")
    # Recalculate width and height
    adjusted_x = x - 10
    adjusted_y = y - 10
    adjusted_width = 40
    adjusted_height = 40

    bet_allowed_coordinates = (adjusted_x, adjusted_y, adjusted_width, adjusted_height)
    # Return the adjusted game window's coordinates and size
    return (adjusted_x, adjusted_y, adjusted_width, adjusted_height)

def get_empty_history_coordinates():
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/empty_history.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.7:
        x, y = max_loc
        return(x, y, 300, 60)
    else:
        print("Could not locate empty history button with sufficient confidence.")
        return None

def get_10_bet_coordinates():
    global bet_10_coordinates
    if bet_10_coordinates[0] is not None:
        return bet_10_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_10_coordinates = (bet_5000_coordinates[0] - 290, bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    return bet_10_coordinates

def get_50_bet_coordinates():
    global bet_50_coordinates
    if bet_50_coordinates[0] is not None:
        return bet_50_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_50_coordinates = (bet_5000_coordinates[0] - 230, bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    screenshot = pyautogui.screenshot(region=bet_50_coordinates)
    screenshot.save("./assets/screenshots/50_btn.png")
    return bet_50_coordinates

def get_250_bet_coordinates():
    global bet_250_coordinates
    if bet_250_coordinates[0] is not None:
        return bet_250_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_250_coordinates = (bet_5000_coordinates[0] - 170, bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    screenshot = pyautogui.screenshot(region=bet_250_coordinates)
    screenshot.save("./assets/screenshots/250_btn.png")
    return bet_250_coordinates

def get_500_bet_coordinates():
    global bet_500_coordinates
    if bet_500_coordinates[0] is not None:
        return bet_500_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_500_coordinates = (bet_5000_coordinates[0] - 110, bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    screenshot = pyautogui.screenshot(region=bet_500_coordinates)
    screenshot.save("./assets/screenshots/500_btn.png")
    return bet_500_coordinates

def get_1000_bet_coordinates():
    global bet_1000_coordinates
    if bet_1000_coordinates[0] is not None:
        return bet_1000_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_1000_coordinates = (bet_5000_coordinates[0] - 50, bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    return bet_1000_coordinates

def get_2000_bet_coordinates():
    global bet_2000_coordinates
    if bet_2000_coordinates[0] is not None:
        return bet_2000_coordinates
    
    global bet_5000_coordinates
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()
    bet_2000_coordinates = (bet_5000_coordinates[0], bet_5000_coordinates[1], bet_5000_coordinates[2], bet_5000_coordinates[3])
    return bet_2000_coordinates

def get_5000_bet_coordinates():
    global bet_5000_coordinates
    if bet_5000_coordinates[0] is not None:
        return bet_5000_coordinates
    
    
    while bet_5000_coordinates is None:
        bet_5000_coordinates = get_bet_allowed_coordinates()

    return bet_5000_coordinates

def get_banker_bet_coordinates():
    global banker_bet_coordinates
    if banker_bet_coordinates[0] is not None:
        return banker_bet_coordinates
    coordinates = get_2000_bet_coordinates()
    coordinates = (coordinates[0] - 30, coordinates[1] - 170, coordinates[2] + 90, coordinates[3] + 80)
    screenshot = pyautogui.screenshot(region=coordinates)
    screenshot.save("./assets/screenshots/banker_btn.png")
    banker_bet_coordinates = coordinates
    return banker_bet_coordinates
    
def get_player_bet_coordinates():
    global player_bet_coordinates
    if player_bet_coordinates[0] is not None:
        return player_bet_coordinates
    coordinates = get_10_bet_coordinates()
    coordinates = (coordinates[0] - 30, coordinates[1] - 170, coordinates[2] + 90, coordinates[3] + 80)
    screenshot = pyautogui.screenshot(region=coordinates)
    screenshot.save("./assets/screenshots/player_btn.png")
    player_bet_coordinates = coordinates
    return player_bet_coordinates

def get_lobby_btn_coordinates():
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/corner_right.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.8:
        x, y = max_loc
        return(x, y, 40, 30)
    else:
        print("Could not locate lobby button with sufficient confidence.")
        return None
    
def get_close_running_game_coordinates():
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/close_running_game.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.65:
        x, y = max_loc
        return (x+65, y+15, 12, 12)
    else:
        print("Could not locate close running game button with sufficient confidence.")
        return None

def ensure_directory_exists(file_path):
    """Ensure the directory exists for a given file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
    return file_path

def capture_full_screen(save_path="./assets/screenshots/full_screen.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return save_path

def capture_coordinates(coordinates, save_path="./assets/screenshots/captured_area.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=coordinates)
    screenshot.save(save_path)
    return save_path

def capture_nameless_window(save_path="./assets/screenshots/nameless_window.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=get_nameless_window_coordinates())
    screenshot.save(save_path)
    return save_path

def capture_nameless_betbox(save_path="./assets/screenshots/bet_area.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=get_nameless_bet_coordinates())
    screenshot.save(save_path)
    return save_path

def capture_nameless_cubes(save_path="./assets/screenshots/cubes_area.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=get_nameless_cube_coordinates())
    screenshot.save(save_path)
    return save_path

def capture_game_window(save_path="./assets/screenshots/game_window.png"):
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=get_game_window_coordinates())
    screenshot.save(save_path)
    return save_path

def capture_game_result(save_path="./assets/screenshots/game_result.png"):
    coordinates = get_game_result_coordinates()
    if coordinates is None:
        return None
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=coordinates)
    screenshot.save(save_path)
    return save_path

def capture_bet_allowed(save_path="./assets/screenshots/bet_allowed.png"):
    coordinates = get_bet_allowed_coordinates()
    if coordinates is None:
        return None
    save_path = ensure_directory_exists(save_path)
    screenshot = pyautogui.screenshot(region=coordinates)
    screenshot.save(save_path)
    return save_path

def crop_history_box(image_path, crop_coordinates, save_path="./assets/screenshots/cropped_history_box.png"):
    img = Image.open(image_path)
    cropped_img = img.crop(crop_coordinates)
    save_path = ensure_directory_exists(save_path)
    cropped_img.save(save_path)
    return cropped_img

def capture_history(running, threshold=0.8):
    if running:
        game_window = capture_game_window()
    else:
        game_window = "./assets/screenshots/game_window.png"
    cropped_image = crop_history_box(game_window, (11, 775, 530, 900))
    outcomes = table_history_template_match(cropped_image)
    return outcomes