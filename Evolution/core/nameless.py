import logging
from core.interaction import click_button
from core.screencapture import capture_full_screen
import cv2
import pyautogui

def get_win_btn_location(threshold=0.8):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_win.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        x = x + 5
        y = y + 5
        width = 60
        height = 30

        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot.save("./assets/screenshots/win_btn.png")
        # Return the adjusted game window's coordinates and size
        return (x, y, width, height)
    else:
        logging.debug("Could not locate win button with sufficient confidence.")
        return None

def get_tie_btn_location(threshold=0.65):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_tie.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        x = x + 10
        y = y + 5
        width = 40
        height = 30

        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot.save("./assets/screenshots/tie_btn.png")
        # Return the adjusted game window's coordinates and size
        return (x, y, width, height)
    else:
        logging.debug("Could not locate tie button with sufficient confidence.")
        return None
    
def get_loss_btn_location(threshold=0.8):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_loss.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc

        x = x + 5
        y = y + 5
        width = 60
        height = 30

        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot.save("./assets/screenshots/loss_btn.png")
        # Return the adjusted game window's coordinates and size
        return (x, y, width, height)
    else:
        logging.debug("Could not locate loss button with sufficient confidence.")
        return None
    
def get_player_start_btn_location(threshold=0.65):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_playerstart.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
    else:
        logging.debug("Could not locate player start button with sufficient confidence.")
        return None
    # Return the adjusted game window's coordinates and size
    x = x + 5
    y = y + 5
    width = 100
    height = 30
    return (x, y, width, height)

def get_player_only_btn_location(threshold=0.65):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_player.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
    else:
        reference = cv2.imread("./assets/templates/nameless_btn_player_pressed.png")
        result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            x, y = max_loc
        else:
            logging.debug("Could not locate player only button with sufficient confidence.")
            return None
    # Return the adjusted game window's coordinates and size
    x = x + 5
    y = y + 5
    width = 100
    height = 30
    return (x, y, width, height)

def get_banker_start_btn_location(threshold=0.65):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_bankerstart.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
    else:
        logging.debug("Could not locate banker start button with sufficient confidence.")
        return None
    # Return the adjusted game window's coordinates and size
    x = x + 5
    y = y + 5
    width = 100
    height = 30
    return (x, y, width, height)

def get_banker_only_btn_location(threshold=0.65):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_banker.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
    else:
        reference = cv2.imread("./assets/templates/nameless_btn_banker_pressed.png")
        result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            x, y = max_loc
        else:
            logging.debug("Could not locate banker only button with sufficient confidence.")
            return None
    # Return the adjusted game window's coordinates and size
    x = x + 5
    y = y + 5
    width = 100
    height = 30
    return (x, y, width, height)

def get_new_line_btn_location(threshold=0.7):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_startnewline.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
        x = x + 5
        y = y + 5
        width = 40
        height = 20
        # Return the adjusted game window's coordinates and size
        return (x, y, width, height)
    else:
        logging.debug("Could not locate new line button with sufficient confidence.")
        return None
    
def get_end_line_btn_location(threshold=0.8):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_endline.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        reference = cv2.imread("./assets/templates/nameless_btn_endline_loss.png")

        result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            logging.debug("Could not locate end line button with sufficient confidence.")
            return None

    x, y = max_loc
    x = x + 5
    y = y + 5
    width = 60
    height = 25
    # Return the adjusted game window's coordinates and size
    return (x, y, width, height)
    
def get_reduce_btn_location(threshold=0.8):
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_reduce.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
        x = x + 5
        y = y + 5
        width = 60
        height = 25

        # Return the adjusted game window's coordinates and size
        return (x, y, width, height)
    else:
        logging.debug("Could not locate reduce button with sufficient confidence.")
        return None

def is_line_done():
    screen = cv2.imread(capture_full_screen())
    reference = cv2.imread("./assets/templates/nameless_btn_resume.png")

    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, _ = cv2.minMaxLoc(result)

    if max_val >= 0.8:
        return True
    else:
        return False

def press_win_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_win_btn_location()
    else:
        coordinates = get_win_btn_location()
    click_button(coordinates)

def press_tie_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_tie_btn_location()
    else:
        coordinates = get_tie_btn_location()
    click_button(coordinates)

def press_loss_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_loss_btn_location()
    else:
        coordinates = get_loss_btn_location()
    click_button(coordinates)

def press_player_start_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_player_start_btn_location()
    else:
        coordinates = get_player_start_btn_location()
    click_button(coordinates)

def press_player_only_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_player_only_btn_location()
    else:
        coordinates = get_player_only_btn_location()
    click_button(coordinates)

def press_banker_start_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_banker_start_btn_location()
    else:
        coordinates = get_banker_start_btn_location()
    click_button(coordinates)

def press_banker_only_btn(retry_until_success=False):
    coordinates = None
    if retry_until_success:
        while coordinates is None:
            coordinates = get_banker_only_btn_location()
    else:
        coordinates = get_banker_only_btn_location()
    click_button(coordinates)

def press_new_line_btn():
    click_button(get_new_line_btn_location())

def press_end_line_btn():
    click_button(get_end_line_btn_location())

def press_reduce_btn():
    click_button(get_reduce_btn_location())