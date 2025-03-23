import logging
from core.analysis import lobby_history_template_match
import cv2
import numpy as np
import pyautogui
from PIL import Image
import time
import random
from core.strategy import analyze_first_6, analyze_first_6_rule, analyze_bias, get_outcomes_without_not_played, get_outcomes_without_ties
from core.interaction import scroll_lobby
from core.ocr import lobby_is_speed_baccarat

def find_lobby_bias(min_bias=4, bias_threshold=65):
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
    bias = evaluate_table_bias(table_histories, min_bias, bias_threshold)
    follow_rule = evaluate_first_6_rule(table_histories)
    
    if not follow_rule:
        exit("The tables are not following the rules. Exiting...")
    
    #bad_cadence = detect_bad_cadence(table_histories)
    #if bad_cadence:
    #    exit("Bad cadence detected. Exiting...")

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
        region_including_name = screenshot[y+h:y + h + 28, x-5:x + w]
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
    x_offset = 0
    y_offset = 0

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x_offset = round(col * 0.2)
            y_offset = round(row * 0.3)
            x1, y1 = col * cell_width - x_offset, row * cell_height + y_offset
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

def detect_bad_cadence(table_histories, min_history=20):
    """
    Detect bad cadence patterns based on actual bet results from bet manager.
    
    Args:
    - table_histories: The histories of different tables for analysis
    
    Returns:
    - (bool): Bad cadence detected
    """
    bad_cadence_result = []
    for box, history in table_histories:
        played_history = get_outcomes_without_not_played(history)
        if len(played_history) < min_history:
            continue
        mode = analyze_first_6(played_history, "", 100)
        if mode != "Skip" and mode != "Analyzing":
            num_losses, bad_cadence = analyze_bad_cadence(played_history, mode)
            bad_cadence_result.append((num_losses, bad_cadence))
    
    if len(bad_cadence_result) == 0:
        return False
    
    number_of_bad_cadence = 0
    for num_losses, bad_cadence in bad_cadence_result:
        if bad_cadence:
            number_of_bad_cadence += 1

    logging.info(f"Bad cadence detected in {number_of_bad_cadence} out of {len(bad_cadence_result)} tables.")
    if number_of_bad_cadence >= len(bad_cadence_result) / 2:
        return True
    return False

def analyze_bad_cadence(history, mode):
    """
    Analyze bad cadence patterns based on history of the table.
    
    Args:
    - history: The history of a single table for analysis
    - mode: The mode of the table (P or B)
    
    Returns:
    - (int, bool): Number of losses, bad cadence detected
    """
    if mode == "PPP":
        return analyze_bad_cadence_banker(history)
    elif mode == "BBB":
        return analyze_bad_cadence_player(history)
    else:
        raise ValueError("Invalid mode. Expected 'PPP' or 'BBB'.")
    
def analyze_bad_cadence_banker(history):
    """
    Analyze bad cadence patterns based on history of the table (Banker mode).
    
    Args:
    - history: The history of a single table for analysis
    
    Returns:
    - (int, bool): Number of losses, bad cadence detected
    """
    first_6 = history[:6]
    number_of_analyzing_games = 6
    if first_6.count("T") > 0:
        number_of_analyzing_games = 7

    bet_outcomes = []
    num_outcomes = len(history)
    bad_cadence = False
    switching = False
    current = "B"
    for i, outcome in enumerate(history):
        if i < number_of_analyzing_games:
            continue

        if switching:
            if current == "B":
                current = "P"
            else:
                current = "B"

        no_ties = get_outcomes_without_ties(history[:i])
        if len(no_ties) >= 6 and no_ties[i-6:i].count("PPP") > 0:
            if not switching:
                switching = True
                current = "P"
        else:
            switching = False
            current = "B"

        if outcome == current:
            bet_outcomes.append("W")
        elif outcome == "T":
            bet_outcomes.append("T")
            continue
        else:
            bet_outcomes.append("L")
    
    num_losses, num_wins, bad_cadence = analyze_cadence_outcomes(bet_outcomes, 6, 12, 2)

    if not bad_cadence:
        # if number of losses is greater than 65% of total games, it is bad cadence
        if num_losses > num_outcomes * 0.65:
            bad_cadence = True

    return num_losses, bad_cadence

def analyze_bad_cadence_player(history):
    """
    Analyze bad cadence patterns based on history of the table (Player mode).
    
    Args:
    - history: The history of a single table for analysis
    
    Returns:
    - (int, bool): Number of losses, bad cadence detected
    """
    first_6 = history[:6]
    number_of_analyzing_games = 6
    if first_6.count("T") > 0:
        number_of_analyzing_games = 7

    bet_outcomes = []
    num_outcomes = len(history)
    bad_cadence = False
    switching = False
    current = "P"
    for i, outcome in enumerate(history):
        if i < number_of_analyzing_games:
            continue

        if switching:
            if current == "B":
                current = "P"
            else:
                current = "B"

        no_ties = get_outcomes_without_ties(history[:i])
        if len(no_ties) >= 6 and no_ties[i-6:i].count("BBB") > 0:
            if not switching:
                switching = True
                current = "B"
        else:
            switching = False
            current = "P"

        if outcome == current:
            bet_outcomes.append("W")
        elif outcome == "T":
            bet_outcomes.append("T")
            continue
        else:
            bet_outcomes.append("L")
    
    num_losses, num_wins, bad_cadence = analyze_cadence_outcomes(bet_outcomes, 6, 12, 2)

    if not bad_cadence:
        # if number of losses is greater than 65% of total games, it is bad cadence
        if num_losses > num_outcomes * 0.65:
            bad_cadence = True

    return num_losses, bad_cadence

def analyze_cadence_outcomes(bet_outcomes, bad_cadence_threshold=6, max_consecutive_loss_threshold=12, bad_streak_threshold=2):
    """
    Analyze the outcomes of the bets to detect bad cadence patterns.
    
    Args:
    - bet_outcomes: List of bet outcomes (W, L, T)
    
    Returns:
    - (int, bool): Number of losses, number of wins, bad cadence detected
    """
    num_losses = 0
    max_consecutive_loss = 0
    current_consecutive_loss = 0
    num_wins = 0
    max_consecutive_win = 0
    current_consecutive_win = 0
    bad_cadence = False
    logging.info(f"Analyzing cadence outcomes: {bet_outcomes}")
    bad_streak = 0
    for outcome in bet_outcomes:
        if outcome == "L":
            num_losses += 1
            current_consecutive_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, current_consecutive_loss)
            current_consecutive_win = 0
            if max_consecutive_loss == bad_cadence_threshold:
                bad_streak += 1
        elif outcome == "W":
            num_wins += 1
            current_consecutive_win += 1
            max_consecutive_win = max(max_consecutive_win, current_consecutive_win)
            current_consecutive_loss = 0
    
    if (max_consecutive_loss >= max_consecutive_loss_threshold) and (num_wins < num_losses):
        bad_cadence = True
    
    if bad_streak >= bad_streak_threshold:
        bad_cadence = True
    
    return num_losses, num_wins, bad_cadence