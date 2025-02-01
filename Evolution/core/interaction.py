# Handles clicking buttons and simulating mouse interactions.

import pyautogui
import random
import time
import math
import logging

def human_like_mouse_move(target_x, target_y, duration=0.5):
    """
    Simulate human-like mouse movement to a target position.
    """
    current_x, current_y = pyautogui.position()
    steps = random.randint(3, 10)  # Number of intermediate steps
    duration = duration / (steps + 1)
    for i in range(steps):
        x = current_x + (target_x - current_x) * (i / steps) + random.uniform(-4, 4)
        y = current_y + (target_y - current_y) * (i / steps) + random.uniform(-4, 4)
        pyautogui.moveTo(x, y, duration)
    pyautogui.moveTo(target_x, target_y, duration)

def click_button(button_location):
    """
    Simulate a mouse click on the specified button.
    """
    current_x, current_y = pyautogui.position()
    #offset_x = random.randint(-(button_location[2] // 2)+10, (button_location[2] // 2)-10)
    x_press = random.randint(button_location[0]+10, (button_location[0] + button_location[2])-10)
    y_press = random.randint(button_location[1]+10, (button_location[1] + button_location[3])-10)
    #offset_y = random.randint(-(button_location[3] // 2)-10, (button_location[3] // 2)-10)
    #button_center = ((button_location[0] + button_location[2] // 2), (button_location[1] + button_location[3] // 2))
    #button_press = ((button_location[0] + button_location[2] // 2) + offset_x, (button_location[1] + button_location[3] // 2) + offset_y)
    #distance = math.sqrt((current_x - button_center[0]) ** 2 + (current_y - button_center[1]) ** 2)
    if current_x < button_location[0] or current_x > (button_location[0]+button_location[2]) or current_y < button_location[1] or current_y > (button_location[1] + button_location[3]):
        human_like_mouse_move(x_press, y_press, duration=random.uniform(0.05, 0.1))
    else:
        logging.info(f"Current position is within the button area. Skipping mouse movement. Current position: ({current_x}, {current_y}), Button area: (X={button_location[0]}, {button_location[0] + button_location[2]}. Y={button_location[1]}, {button_location[1] + button_location[3]})")
    time.sleep(random.uniform(0.04, 0.08))
    pyautogui.click()

def random_mouse_move(probability=0.001):
    """
    Randomly move the mouse near the current position.
    """
    if random.random() < probability:
        random_x = random.randint(pyautogui.position()[0] - 5, pyautogui.position()[0] + 5)
        random_y = random.randint(pyautogui.position()[1] - 5, pyautogui.position()[1] + 5)
        pyautogui.moveTo(random_x, random_y, duration=random.uniform(0.5, 1.5))

def random_mouse_placement(screen_width, screen_height):
    """
    Randomly move the mouse to a random position on the screen.
    """
    random_x = random.randint(50, screen_width)
    random_y = random.randint(100, screen_height)
    pyautogui.moveTo(random_x, random_y, duration=random.uniform(0.4, 0.8))