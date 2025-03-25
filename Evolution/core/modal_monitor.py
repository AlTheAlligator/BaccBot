import threading
import time
import pyautogui
import logging
import random
from core.interaction import click_button
import cv2
from core.screencapture import capture_full_screen

def monitor_modal(poll_interval: float = 0.1, confidence: float = 0.8):
    """
    Continuously monitors the screen for the modal dialog and dismisses it when found.
    
    :param modal_template: File path to an image template of the modal (optional; can be used for logging).
    :param button_template: File path to an image template of the modal's dismiss button.
    :param poll_interval: How frequently to check for the modal (in seconds).
    :param confidence: Confidence threshold for image matching (0 to 1).
    """
    while True:
        try:
            screen = cv2.imread(capture_full_screen("./assets/screenshots/full_screen_modal.png"))
            reference = cv2.imread("./assets/templates/jeg_forstar_btn.png")
            #reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)

            result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= confidence:
                logging.info("Modal detected on screen, removing...")
                x, y = max_loc
                x = x + 2
                y = y + 2
                w = 10
                h = 10
                time.sleep(random.uniform(0.3, 0.6))
                click_button((x, y, w, h))
                time.sleep(random.uniform(0.1, 0.3))

                screen = cv2.imread(capture_full_screen("./assets/screenshots/full_screen_modal.png"))
                reference = cv2.imread("./assets/templates/fortsat_med_spil.png")

                result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val >= confidence:
                    logging.info("Continue playing detected on screen, removing...")
                    x, y = max_loc
                    w = 60
                    h = 30
                    click_button((x, y, w, h))
                else:
                    logging.info("Continue playing not detected on screen.")

            # Sleep for a short interval before checking again.
            time.sleep(poll_interval)
        except Exception as e:
            # Log the exception and continue; you might want to add more robust error handling.
            logging.error(f"Error in modal monitor: {e}")
            time.sleep(poll_interval)

def start_modal_monitor_thread():
    """
    Starts the modal monitor thread as a daemon.
    
    :param modal_template: File path to the modal's template image.
    :param button_template: File path to the dismiss button's template image.
    """
    monitor_thread = threading.Thread(
        target=monitor_modal,
        daemon=True  # Daemon thread ensures it won't block program exit.
    )
    monitor_thread.start()
    logging.info("Modal monitor thread started.")
    return monitor_thread