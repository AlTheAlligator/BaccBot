import logging
from core.analysis import lobby_history_template_match
import cv2
import numpy as np
import pyautogui
from PIL import Image
import time
import random
from core.strategy import analyze_first_6, analyze_first_6_rule, analyze_bias, get_outcomes_without_not_played
from core.interaction import scroll_lobby
from core.ocr import extract_bet_size, preprocess_image, lobby_is_speed_baccarat

def find_lobby_bias(consecutive_checks=3, bias_threshold=65):
    """Analyzes the lobby to determine the current bias."""
    logging.info("Analyzing the lobby for bias...")

    # First analysis pass
    scroll = random.randint(3, 4)
    for i in range(scroll):
        scroll_lobby("down")
        time.sleep(random.uniform(0.1, 0.15))

    time.sleep(random.uniform(3, 8))
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)[:, :, ::-1]
    table_boxes = find_tables(screenshot)
    table_histories = extract_table_histories(screenshot, table_boxes, False)

    # Second analysis pass
    scroll = random.randint(8, 10)
    for i in range(scroll):
        scroll_lobby("down")
        time.sleep(random.uniform(0.11, 0.2))

    time.sleep(random.uniform(3, 8))
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)[:, :, ::-1]
    table_boxes = find_tables(screenshot)
    table_histories_2 = extract_table_histories(screenshot, table_boxes, False)
    table_histories.extend(table_histories_2)

    # Natural scrolling back to top
    for i in range(20):
        scroll_lobby("up")
        time.sleep(random.uniform(0.05, 0.11))

    # Analyze bias
    bias = evaluate_table_bias(table_histories, consecutive_checks, bias_threshold)
    follow_rule = evaluate_first_6_rule(table_histories)

    if not follow_rule:
        exit("The tables are not following the rules. Exiting...")

    if bias:
        logging.info(f"Detected bias: {bias}")
        return bias

    logging.info("No bias found.")
    return ""

def evaluate_table_bias(table_histories, min_bias=3, bias_threshold=65):
    # Evaluate histories to find a suitable table
    biases = []
    for box, history in table_histories:
        biases.append(analyze_bias(history, 24, 65))

    biases = [b for b in biases if b is not None]
    logging.info(f"Biases: {biases}")
    player_or_banker = [b for b in biases if b in ["P", "B"]]
    if len(player_or_banker) < min_bias:
        logging.info(f"Not enough biases to analyze, only {len(player_or_banker)} was found.")
        return ""
    
    player_count = player_or_banker.count('P')
    banker_count = player_or_banker.count('B')
    if player_count >= len(player_or_banker) / 100 * bias_threshold:
        logging.info("Player lobby bias detected.")
        return "P"
    elif banker_count >= len(player_or_banker) / 100 * bias_threshold:
        logging.info("Banker lobby bias detected.")
        return "B"
    else:
        logging.info("No lobby bias detected.")
        return ""

def evaluate_first_6_rule(table_histories):
    rule_analysis = []
    for box, history in table_histories:
        rule_analysis.append(analyze_first_6_rule(history))
    rule_analysis = [r for r in rule_analysis if r[0] != "Skip"]
    if len(rule_analysis) == 0:
        logging.info("No rule analysis found.")
        return True
    
    following_rule = [r for r in rule_analysis if r[1] is True]
    if len(following_rule) >= len(rule_analysis) / 2:
        logging.info(f"The tables are following the rules. {len(following_rule)} out of {len(rule_analysis)} tables are following the rules.")
        return True
    
    logging.info(f"The tables are not following the rules. {len(following_rule)} out of {len(rule_analysis)} tables are following the rules.")
    return False

def find_tables(screenshot):
    """Identifies all table regions from the lobby screenshot."""
    logging.info("Detecting tables in the lobby...")
    table_bounding_boxes = []

    # Preprocessing
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 350, 650)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Filter by aspect ratio and size
        if 5 < aspect_ratio < 6 and 150 < w < 500 and 50 < h < 100:
            table_bounding_boxes.append((x, y, w, h))

    # Apply non-maximum suppression
    table_bounding_boxes = non_maximum_suppression(table_bounding_boxes)
    save_debug_image_with_boxes(screenshot, table_bounding_boxes)
    
    logging.info(f"Found {len(table_bounding_boxes)} potential tables.")
    return table_bounding_boxes

def non_maximum_suppression(boxes, overlap_threshold=0.5):
    """Apply non-maximum suppression to remove overlapping boxes"""
    if not boxes:
        return []

    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    
    # Extract coordinates
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    # Compute areas
    areas = w * h
    
    # Sort by width (larger tables first)
    indices = np.argsort(w)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x[i], x[indices[1:]])
        yy1 = np.maximum(y[i], y[indices[1:]])
        xx2 = np.minimum(x[i] + w[i], x[indices[1:]] + w[indices[1:]])
        yy2 = np.minimum(y[i] + h[i], y[indices[1:]] + h[indices[1:]])
        
        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        overlap = (w_inter * h_inter) / areas[indices[1:]]
        
        # Remove overlapping boxes
        indices = indices[1:][overlap < overlap_threshold]
    
    return boxes[keep].tolist()

def save_debug_image_with_boxes(screenshot, table_boxes, output_path="./assets/screenshots/debug_detected_tables.png"):
    """Save debug image with detected table boxes"""
    debug_image = screenshot.copy()
    for (x, y, w, h) in table_boxes:
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_path, debug_image)

def extract_table_histories(screenshot, table_boxes, check_for_speed_baccarat=True):
    """Extract game histories from detected tables"""
    logging.info("Extracting table histories...")
    table_histories = []

    for box in table_boxes:
        x, y, w, h = box
        region_including_name = screenshot[y+h:y + h + 32, x-5:x + w]
        if check_for_speed_baccarat:
            if not lobby_is_speed_baccarat(region_including_name):
                continue
        table_region = screenshot[y+3:y + h, x+1:x + w]

        img = Image.fromarray(table_region)
        img.save("./assets/screenshots/table.png".format(x, y))
        outcomes = history_template_match(table_region, (6, 34))
        table_histories.append((box, outcomes))

    return table_histories

def match_on_color(cell, target_colors=None):
    """Match cell color to determine outcome"""
    if target_colors is None:
        target_colors = {
            'P': (0, 0, 255),   # Red
            'B': (255, 0, 0),   # Blue
            'T': (0, 255, 0),   # Green
            'N': (255, 255, 255) # White
        }
    
    # Calculate average color of cell
    avg_color = np.mean(cell, axis=(0, 1))
    
    # Find closest matching color
    min_dist = float('inf')
    result = None
    
    for outcome, color in target_colors.items():
        dist = np.sum((avg_color - np.array(color)) ** 2)
        if dist < min_dist:
            min_dist = dist
            result = outcome
            
    return result

TARGET_COLORS = {
        "B": (108, 169, 255),     # Blue (Inverted because of Canny edge detection)
        "P": (245, 138, 138),     # Red (Inverted because of Canny edge detection)
        "T": (74, 171, 118),     # Green
        "N": (255, 255, 255)  # White
    }
def history_template_match(cropped_img, grid_size=(6, 36)):
    img_array = np.array(cropped_img)
    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]
    offset = 0

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x1, y1 = (col + offset) * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cell = img_array[y1:y2, x1:x2]

            img = Image.fromarray(cell)
            img.save("./assets/screenshots/lobby_cells/cell_{}_{}.png".format(row, col))

            detected = match_on_color(cell, target_colors=TARGET_COLORS)
            outcomes.append(detected)
    return outcomes

def evaluate_table_histories(table_histories, bias):
    """
    Evaluate table histories to find suitable table.
    Returns the coordinates of the first suitable table found.
    """
    # Evaluate histories to find a suitable table
    for box, history in table_histories:
        logging.info(f"Analyzing table at {box} with history: {get_outcomes_without_not_played(history)}")
        if analyze_first_6(history, bias) not in ["Skip", "Analyzing"]:
            logging.info(f"Found suitable table at {box} with history: {get_outcomes_without_not_played(history)}")
            return box
    
    logging.info("No suitable table found.")
    return None