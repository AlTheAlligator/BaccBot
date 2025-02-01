import logging
from pynput import keyboard
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
stop_event = threading.Event()

def log(message, level="info"):
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)

def on_press(key):
    try:
        # Check if the key pressed is the escape key
        if key == keyboard.Key.esc:
            print("ESC key pressed. Stopping the program...")
            stop_event.set()
            return False  # Stop the listener
    except Exception as e:
        print(f"Error in keyboard listener: {e}")

def start_keyboard_listener():
    # Start the listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()