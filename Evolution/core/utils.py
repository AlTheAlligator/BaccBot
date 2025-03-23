from datetime import datetime, timedelta
from pynput import keyboard
import threading
import logging

stop_event = threading.Event()
start = datetime.now()

def on_press(key):
    try:
        # Check if the key pressed is the F12 key
        if key == keyboard.Key.f12:
            print("F12 key pressed. Stopping the program...")
            stop_event.set()
            return False  # Stop the listener
        #if datetime.now() - start > timedelta(hours=4):
        #    print("Stopping the keyboard listener after 4 hours.")
        #    stop_event.set()
        #    return False  # Stop the listener
    except Exception as e:
        print(f"Error in keyboard listener: {e}")

def start_keyboard_listener():
    # Start the listener in a separate thread
    logging.info("Starting keyboard listener, press F12 to stop the program...")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()