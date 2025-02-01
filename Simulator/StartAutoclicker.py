import pyautogui
import cv2
import numpy as np
import time
import random
from PIL import Image
import logging

result_templates = {
    "B": "Screenshots/banker.png",  # Red circle template
    "P": "Screenshots/player.png",  # Blue circle template
    "T": "Screenshots/tie.png",     # Green circle template
    "N": "Screenshots/blank.png"    # Blank circle template
}

# Capture the entire screen
def capture_full_screen():
    screenshot = pyautogui.screenshot()
    screenshot.save("Screenshots/full_screen.png")  # Save for debugging
    return screenshot

def locate_game_window(screen_path, reference_path):
    # Load the full screen and reference images
    screen = cv2.imread(screen_path)
    reference = cv2.imread(reference_path)

    # Perform template matching
    result = cv2.matchTemplate(screen, reference, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Define a threshold for matching
    threshold = 0.8  # Adjust based on accuracy
    if max_val >= threshold:
        return max_loc  # Top-left corner of the reference image
    else:
        return None  # Reference image not found

def get_game_window_coordinates():
    # Step 1: Capture the full screen
    full_screen = capture_full_screen()
    full_screen_path = "./Screenshots/full_screen.png"

    # Step 2: Define the reference image
    reference_path = "./Screenshots/GameScreen.png"  # Save your reference image beforehand

    # Step 3: Locate the reference image
    location = locate_game_window(full_screen_path, reference_path)
    if location:
        x, y = location
        print(f"Reference image found at: {x}, {y}")
        # Define the game window relative to the reference image
        game_window = (x, y, 800, 600)  # Adjust width and height based on your game
        return game_window
    else:
        print("Reference image not found.")
        return None

def capture_game_window(game_window_coordinates):
    screenshot = pyautogui.screenshot(region=game_window_coordinates)
    screenshot.save("Screenshots/game_window.png")  # Save for debugging
    return screenshot

def crop_history_box(image_path, crop_coordinates):
    # Load the image
    img = Image.open(image_path)
    # Crop the history box (define coordinates based on the red box)
    cropped_img = img.crop(crop_coordinates)
    cropped_img.save("Screenshots/cropped_history_box.png")  # Save for debugging
    return cropped_img

def template_match(cropped_img, templates, grid_size=(6, 14)):
    """
    Perform template matching for each cell in the grid to detect outcomes.
    
    Args:
    - cropped_img: Cropped history box (PIL Image).
    - templates: Dictionary of template images (key: outcome, value: file path).
    - grid_size: Number of rows and columns in the grid.
    
    Returns:
    - outcomes: List of detected outcomes (e.g., ["B", "P", "T", ...]).
    """
    # Convert the cropped image to a numpy array (OpenCV format)
    img_array = np.array(cropped_img)

    # Get cell dimensions
    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            # Extract each cell
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cell = img_array[y1:y2, x1:x2]
            Image.fromarray(cell).save(f"Screenshots/cells/cell_{row}_{col}.png")  # Save for debugging

            # Validate cell dimensions
            if cell.shape[0] == 0 or cell.shape[1] == 0:
                raise ValueError(f"Cell size is invalid: {cell.shape}")

            # Convert cell to grayscale
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Match templates
            detected = "Unknown"
            for outcome, template_path in templates.items():
                # Load and validate the template
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is None:
                    raise ValueError(f"Template image {template_path} could not be loaded.")

                # Convert template to grayscale
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                # Perform template matching
                result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > 0.8:  # Match threshold
                    detected = outcome
                    break
            
            outcomes.append(detected)
    
    return outcomes

def locate_button(game_window_image, button_template_path):
    """
    Locate a button in the game window using template matching.
    
    Args:
    - game_window_image: Path to the screenshot of the game window.
    - button_template_path: Path to the button template image.
    
    Returns:
    - button_location: Top-left corner (x, y) of the button, or None if not found.
    """
    # Load game window and button template
    game_window = cv2.imread(game_window_image)
    button_template = cv2.imread(button_template_path)

    # Perform template matching
    result = cv2.matchTemplate(game_window, button_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Define a matching threshold
    threshold = 0.8
    if max_val >= threshold:
        return max_loc  # Top-left corner of the button
    else:
        return None  # Button not found

def click_button(button_location, game_window_top_left, button_size=(50, 20)):
    """
    Simulate a mouse click on the specified button, adjusted for the game window's position.

    Args:
    - button_location: Top-left corner (x, y) of the button relative to the game window.
    - game_window_top_left: Top-left corner (x, y) of the game window on the screen.
    - button_size: Width and height of the button for center calculation.
    """
    # Adjust button location relative to the screen
    adjusted_x = button_location[0] + game_window_top_left[0]
    adjusted_y = button_location[1] + game_window_top_left[1]

    # Calculate the center of the button
    button_center = (adjusted_x + button_size[0] // 2, adjusted_y + button_size[1] // 2)

    # Move the mouse and click
    pyautogui.moveTo(button_center[0], button_center[1], duration=random.uniform(0.2, 0.5))
    pyautogui.click()

def interact_with_game(game_window_coordinates):
    
    # Unpack game window coordinates
    game_window_top_left = (game_window_coordinates[0], game_window_coordinates[1])

    # Step 1: Capture game window
    capture_game_window(game_window_coordinates)

    # Step 2: Locate buttons
    game_window_image = "Screenshots/game_window.png"
    buttons = {
        "Deal": "Screenshots/deal.png",
        "Draw": "Screenshots/draw.png",
        "Repeat": "Screenshots/repeat.png"
    }

    for action, template in buttons.items():
        location = locate_button(game_window_image, template)
        if location:
            print(f"{action} button found at: {location}")
            click_button(location, game_window_top_left)
            time.sleep(random.uniform(0.5, 1.5))  # Add delay for realism
        #else:
        #    print(f"{action} button not found!")


def main():
    logging.info("Starting Baccarat outcome detection...")

    # Locate game window
    game_window_coordinates = get_game_window_coordinates()
    if not game_window_coordinates:
        logging.error("Game window not found. Exiting.")
        return

    # Capture game window
    capture_game_window(game_window_coordinates)

    # Unpack game window coordinates
    game_window_top_left = (game_window_coordinates[0], game_window_coordinates[1])
    click_button(locate_button("Screenshots/game_window.png", "Screenshots/history.png"), game_window_top_left)
    time.sleep(0.5)
    # Capture game window
    capture_game_window(game_window_coordinates)

    # Crop history box
    cropped_image = crop_history_box("Screenshots/game_window.png", (22, 22, 375, 175))
    logging.info("Cropped history box saved.")

    # Perform template matching
    outcomes = template_match(cropped_image, result_templates)
    print(f"Detected outcomes: {outcomes[:7]}")  # First 7 outcomes

    while True:
        interact_with_game(game_window_coordinates)

if __name__ == "__main__":
    main()
