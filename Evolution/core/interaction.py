# Handles clicking buttons and simulating mouse interactions.

import pyautogui
import random
import time
import math
import logging
from numpy.random import normal

# Configure PyAutoGUI settings with minimal but natural delays
pyautogui.MINIMUM_SLEEP = 0.01  # Reduced from 0.03-0.05
pyautogui.PAUSE = 0.015  # Reduced from 0.02-0.04

# Human reaction time simulation (200-500ms typical range)
def human_reaction_delay():
    """Simulate minimal but natural reaction time"""
    return max(0.05, abs(normal(0.15, 0.05)))  # Reduced from 300ms to 150ms mean

def simulate_human_fatigue():
    """Simulate occasional brief pauses, but with reduced frequency and duration"""
    if random.random() < 0.02:  # Reduced from 5% to 2% chance
        return random.uniform(0.2, 0.5)  # Reduced from 0.5-2.0s
    return 0

def human_like_mouse_move(target_x, target_y, duration=0.5):
    """
    Simulate human-like mouse movement to a target position.
    """
    # Get the current mouse position as the starting point.
    start_pos = pyautogui.position()
    # Execute the human-like mouse movement.
    human_like_mouse_move_1(start_pos, (target_x, target_y), min_speed=50000, max_speed=120000, speed_variance=0.2, steps=20)

def click_button(button_location):
    """Simulate a more human-like mouse click on the specified button."""
    if button_location is None:
        logging.warning("Button location is None. Skipping click.")
        return

    # Add reaction delay before starting movement
    time.sleep(human_reaction_delay())
    
    current_x, current_y = pyautogui.position()
    
    # Optimize target selection for speed while maintaining natural appearance
    if button_location[2] > 20 and button_location[3] > 20:  # Large button
        edge_buffer_x = int(button_location[2] * 0.15) # Reduced buffer
        edge_buffer_y = int(button_location[3] * 0.15) # Reduced buffer
        x_press = random.randint(button_location[0] + edge_buffer_x, 
                               button_location[0] + button_location[2] - edge_buffer_x)
        y_press = random.randint(button_location[1] + edge_buffer_y,
                               button_location[1] + button_location[3] - edge_buffer_y)
    else:
        x_press = button_location[0] + button_location[2] // 2 + random.randint(-1, 1)
        y_press = button_location[1] + button_location[3] // 2 + random.randint(-1, 1)
    
    # Only move if we're not already in the right spot
    if (current_x < button_location[0] or 
        current_x > (button_location[0] + button_location[2]) or 
        current_y < button_location[1] or 
        current_y > (button_location[1] + button_location[3])):
        
        movement_time = random.uniform(0.1, 0.2)  # Reduced from 0.2-0.4s
        human_like_mouse_move(x_press, y_press, duration=movement_time)
    else:
        logging.debug(f"Current position is within button area")
    
    # Reduce micro-movements
    if random.random() < 0.3:  # 30% chance of micro-movement
        pyautogui.moveRel(random.randint(-1, 1), random.randint(-1, 1), 
                         duration=0.05)
    
    # Add fatigue delay
    time.sleep(simulate_human_fatigue())
    
    # Optimize click timing
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.03, 0.08))  # Reduced from 0.05-0.15s
    pyautogui.mouseUp()

def random_mouse_move(probability=0.0005):  # Reduced probability
    """Minimal idle mouse movement"""
    if random.random() < probability:
        # Humans tend to make larger movements when idle
        radius = random.randint(5, 25)  # Reduced movement range
        angle = random.uniform(0, 2 * math.pi)
        current_x, current_y = pyautogui.position()
        
        target_x = int(current_x + radius * math.cos(angle))
        target_y = int(current_y + radius * math.sin(angle))
        
        # Add natural acceleration/deceleration
        human_like_mouse_move(target_x, target_y, 
                            duration=random.uniform(0.2, 0.4))  # Reduced from 0.3-0.8s

def random_mouse_placement(screen_width, screen_height):
    """
    Randomly move the mouse to a random position on the screen.
    """
    random_x = random.randint(50, screen_width)
    random_y = random.randint(800, screen_height)
    pyautogui.moveTo(random_x, random_y, duration=random.uniform(0.4, 0.8))

def scroll_lobby(direction="down", steps=120):
    """
    Simulate scrolling in the lobby.
"""
    if direction == "down":
        pyautogui.scroll(-steps)
    elif direction == "up":
        pyautogui.scroll(steps)
    else:
        logging.warning(f"Invalid scroll direction: {direction}")
        
def get_cubic_bezier_points(start, control1, control2, end, steps=100):
    """
    Compute points along a cubic Bézier curve.
    
    :param start: Tuple (x, y) for the start point.
    :param control1: Tuple (x, y) for the first control point.
    :param control2: Tuple (x, y) for the second control point.
    :param end: Tuple (x, y) for the end point.
    :param steps: Number of points along the curve.
    :return: List of (x, y) tuples.
    """
    points = []
    for i in range(steps + 1):
        t = i / steps
        # Cubic Bézier formula:
        x = (1 - t) ** 3 * start[0] \
            + 3 * (1 - t) ** 2 * t * control1[0] \
            + 3 * (1 - t) * t ** 2 * control2[0] \
            + t ** 3 * end[0]
        y = (1 - t) ** 3 * start[1] \
            + 3 * (1 - t) ** 2 * t * control1[1] \
            + 3 * (1 - t) * t ** 2 * control2[1] \
            + t ** 3 * end[1]
        points.append((x, y))
    return points

def human_like_mouse_move_1(start, end, min_speed=500, max_speed=1200, speed_variance=0.2, steps=100):
    """
    Moves the mouse pointer from start to end along a smooth, human-like path.
    
    The overall speed is chosen randomly within a given range (min_speed to max_speed),
    and a random variance factor is applied to the duration so that not every move is identical.
    This ensures that longer distances take longer than shorter ones.
    
    :param start: Tuple (x, y) for the starting position.
    :param end: Tuple (x, y) for the ending position.
    :param min_speed: Minimum speed in pixels per second.
    :param max_speed: Maximum speed in pixels per second.
    :param speed_variance: Fraction by which to vary the computed duration (e.g. 0.2 gives 80%-120% of base duration).
    :param steps: Number of interpolation steps (more steps = smoother motion).
    """
    # Compute the distance between start and end.
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    
    # Choose a random base speed within the specified range.
    base_speed = random.uniform(min_speed, max_speed)
    
    # Calculate the base duration (in seconds) for the movement.
    base_duration = distance / base_speed
    
    # Apply a random variance factor to simulate natural variations.
    variance_factor = random.uniform(1 - speed_variance, 1 + speed_variance)
    duration = base_duration * variance_factor

    # Helper function to generate a random control point for the Bézier curve.
    def random_control_point(a, b):
        # Choose a point between a and b (around 30% to 70% of the way), then add a random offset.
        mid = a + (b - a) * random.uniform(0.3, 0.7)
        offset = random.uniform(-100, 100)  # Adjust this offset range to control curve randomness.
        return mid + offset

    start_x, start_y = start
    end_x, end_y = end

    # Generate two control points with randomness.
    control1 = (random_control_point(start_x, end_x), random_control_point(start_y, end_y))
    control2 = (random_control_point(start_x, end_x), random_control_point(start_y, end_y))

    # Get points along the smooth Bézier curve.
    path_points = get_cubic_bezier_points(start, control1, control2, end, steps)
    
    # Calculate delay between each movement step.
    delay = duration / len(path_points)
    
    for point in path_points:
        pyautogui.moveTo(point[0], point[1])
        #time.sleep(delay)