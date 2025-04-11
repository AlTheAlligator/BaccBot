import numpy as np
import cv2
from PIL import Image
import logging
import os

templates = {
    "B": "./assets/templates/history_banker.png",
    "P": "./assets/templates/history_player.png",
    "T": "./assets/templates/history_tie.png",
    "N": "./assets/templates/history_blank.png"
}

# Predefined colors (in BGR format as used by OpenCV)
TABLE_TARGET_COLORS = {
    "P": (108, 169, 255),     # Blue
    "B": (245, 138, 138),     # Red
    "T": (74, 171, 118),     # Green
    "N": (255, 255, 255)  # White
}

# Predefined colors (in BGR format as used by OpenCV)
LOBBY_TARGET_COLORS = {
    "B": (108, 169, 255),     # Blue (Inverted because of Canny edge detection)
    "P": (245, 138, 138),     # Red (Inverted because of Canny edge detection)
    "T": (74, 171, 118),     # Green
    "N": (255, 255, 255)  # White
}

def table_match_on_color(cell, target_colors=TABLE_TARGET_COLORS):
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

def table_history_template_match(cropped_img, grid_size=(6, 26)):
    """Analyze a cropped image of the history table and identify outcomes

    Args:
        cropped_img: PIL Image or numpy array of the cropped history area
        grid_size: Tuple of (rows, columns) in the grid

    Returns:
        List of detected outcomes ('P', 'B', 'T', 'N')
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(cropped_img, Image.Image):
        img_array = np.array(cropped_img)
    else:
        img_array = cropped_img

    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]
    offset = 5

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x1, y1 = col * cell_width + offset, row * cell_height + offset
            x2, y2 = x1 + cell_width - (offset * 2), y1 + cell_height - (offset * 2)
            cell = img_array[y1:y2, x1:x2]

            # Save cell image if in debug mode
            from core.screencapture import get_debug_mode
            if get_debug_mode():
                try:
                    # Ensure directory exists
                    cell_dir = "./assets/screenshots/cells"
                    if not os.path.exists(cell_dir):
                        os.makedirs(cell_dir, exist_ok=True)

                    img = Image.fromarray(cell)
                    img.save("./assets/screenshots/cells/cell_{}_{}.png".format(row, col))
                except Exception as e:
                    logging.error(f"Error saving cell image {row}_{col}: {str(e)}")
                    # Continue execution even if saving fails

            detected = table_match_on_color(cell)
            outcomes.append(detected)
    return outcomes

def lobby_match_on_color(cell, target_colors=LOBBY_TARGET_COLORS):
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

def lobby_history_template_match(cropped_img, grid_size=(6, 26)):
    img_array = np.array(cropped_img)

    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]
    offset = 0

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x1, y1 = col * cell_width + offset, row * cell_height + offset
            x2, y2 = x1 + cell_width - (offset * 2), y1 + cell_height - (offset * 2)
            cell = img_array[y1:y2, x1:x2]

            img = Image.fromarray(cell)
            img.save("./data/screenshots/cells/cell_{}_{}.png".format(row, col))

            detected = lobby_match_on_color(cell)
            outcomes.append(detected)
    return outcomes

def find_tables(self, screenshot):
    """
    Identifies all table regions dynamically from the lobby screenshot.
    :param screenshot: Captured image of the lobby.
    :return: List of bounding boxes for detected tables.
    """
    logging.info("Detecting tables in the lobby...")
    table_bounding_boxes = []

    # Preprocessing
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 350, 650)
    cv2.imwrite("./data/screenshots/edges.png", edges)
    # Find contours (table candidates)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Apply aspect ratio and size filtering
        aspect_ratio = w / float(h)
        if 5 < aspect_ratio < 6 and 150 < w < 500 and 50 < h < 100:  # Adjust these thresholds based on lobby layout
            table_bounding_boxes.append((x, y, w, h))

    table_bounding_boxes = self.non_maximum_suppression(table_bounding_boxes)
    self.save_debug_image_with_boxes(screenshot, table_bounding_boxes)
    logging.info(f"Found {len(table_bounding_boxes)} potential tables.")
    return table_bounding_boxes

def non_maximum_suppression(boxes, overlap_threshold=0.5):
    """
    Perform non-maximum suppression to filter out overlapping bounding boxes.
    :param boxes: List of bounding boxes in (x, y, w, h) format.
    :param overlap_threshold: IoU threshold for filtering.
    :return: Filtered list of bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to a numpy array
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute the area of the boxes
    areas = (x2 - x1) * (y2 - y1)

    # Sort the boxes by their bottom-right y-coordinate
    order = np.argsort(y2)

    filtered_boxes = []
    while len(order) > 0:
        # Select the box with the largest area (last in the sorted list)
        i = order[-1]
        filtered_boxes.append(boxes[i])

        # Compute IoU for all other boxes
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        # Compute width and height of the overlap
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute IoU
        overlap = (w * h) / (areas[order[:-1]] + areas[i] - (w * h))

        # Remove boxes with IoU above the threshold
        order = order[np.where(overlap <= overlap_threshold)[0]]

    return filtered_boxes


def save_debug_image_with_boxes(self, screenshot, table_boxes, output_path="./data/screenshots/debug_detected_tables.png"):
    """
    Saves a debug image with bounding boxes drawn around detected table regions.
    :param screenshot: Captured image of the lobby (numpy array).
    :param table_boxes: List of bounding boxes for detected tables.
    :param output_path: Path to save the debug image.
    """
    debug_image = screenshot.copy()
    for (x, y, w, h) in table_boxes:
        # Draw rectangles around detected tables
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Save the debug image
    cv2.imwrite(output_path, debug_image)
    logging.info(f"Debug image saved: {output_path}")