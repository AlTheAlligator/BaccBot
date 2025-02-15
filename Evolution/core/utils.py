from pynput import keyboard
import threading
import logging

stop_event = threading.Event()

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
    logging.info("Starting keyboard listener, press ESC to stop the program...")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()