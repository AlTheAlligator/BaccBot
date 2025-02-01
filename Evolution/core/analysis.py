import numpy as np
import cv2
from PIL import Image

templates = {
    "B": "./assets/templates/history_banker.png",
    "P": "./assets/templates/history_player.png",
    "T": "./assets/templates/history_tie.png",
    "N": "./assets/templates/history_blank.png"
}

# Predefined colors (in BGR format as used by OpenCV)
TARGET_COLORS = {
    "P": (108, 169, 255),     # Blue
    "B": (245, 138, 138),     # Red
    "T": (74, 171, 118),     # Green
    "N": (255, 255, 255)  # White
}

def match_on_color(cell, target_colors=TARGET_COLORS):
    """
    Match a cell based on its average color.

    Args:
    - cell: The cell image (numpy array).
    - target_colors: Dictionary of target names and their BGR colors.

    Returns:
    - detected: The name of the closest match (e.g., "Player," "Banker," or "Tie").
    """
    # Calculate the average color of the cell
    avg_color = np.mean(cell, axis=(0, 1))  # Average across height and width
    avg_color_bgr = tuple(map(int, avg_color))  # Convert to integers

    # Find the closest target color
    detected = "Unknown"
    min_distance = float("inf")
    for target, color in target_colors.items():
        # Calculate Euclidean distance between colors
        distance = np.sqrt(sum((avg_color_bgr[i] - color[i]) ** 2 for i in range(3)))
        if distance < min_distance:
            min_distance = distance
            detected = target

    return detected

def history_template_match(cropped_img, grid_size=(6, 26)):
    img_array = np.array(cropped_img)

    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]
    offset = 5

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x1, y1 = col * cell_width + offset, row * cell_height + offset
            x2, y2 = x1 + cell_width - (offset * 2), y1 + cell_height - (offset * 2)
            cell = img_array[y1:y2, x1:x2]

            img = Image.fromarray(cell)
            img.save("./assets/screenshots/cells/cell_{}_{}.png".format(row, col))

            detected = match_on_color(cell)
            outcomes.append(detected)
    return outcomes

