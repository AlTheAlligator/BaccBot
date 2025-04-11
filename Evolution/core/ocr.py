import math
import cv2
import pytesseract
import numpy as np
from PIL import Image
from core.analysis import non_maximum_suppression
import logging
import os

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = f'C:\Program Files\Tesseract-OCR\\tesseract.exe'  # Update with your Tesseract path

def preprocess_image(img, use_otsu=False, use_clahe=False, use_inverse=True):
    """
    Common image preprocessing function for OCR
    """
    # Convert to grayscale if image is not already grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if use_clahe:
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if use_otsu:
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif use_inverse:
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
    else:
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
        )

    return thresh

def preprocess_image_v2(img, use_otsu=False, use_clahe=False, use_inverse=True):
    """
    Common image preprocessing function for OCR
    """
    # Convert to grayscale if image is not already grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 85, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def get_game_result(image):
    """
    Extract the game result (BANKER/PLAYER/TIE) from the game result area

    Args:
        image: Either a PIL Image object, OpenCV image, or a path to an image file

    Returns:
        String representing the game result: 'B', 'P', 'T', or 'Waiting for Results'
    """
    # Handle different input types
    if isinstance(image, str):
        # It's a file path
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # It's a PIL Image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Assume it's already an OpenCV image
        img = image

    if img is None:
        return "Waiting for Results"

    # Preprocess with Otsu thresholding
    thresh = preprocess_image(img, use_otsu=True)

    # Save debug image if needed
    from core.screencapture import get_debug_mode
    if get_debug_mode():
        try:
            # Ensure directory exists
            debug_dir = "./assets/screenshots"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)

            Image.fromarray(thresh).save("./assets/screenshots/result_thresholded.png")
        except Exception as e:
            logging.error(f"Error saving result thresholded image: {str(e)}")
            # Continue execution even if saving fails

    # Resize for better OCR accuracy
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    text = pytesseract.image_to_string(resized, config='--psm 6 -c tessedit_char_whitelist=BANKERPLYTIE')
    text = text.strip()

    if text == "BANKER":
        return "B"
    if text == "PLAYER":
        return "P"
    if text == "TIE":
        return "T"

    return "Waiting for Results"

def extract_cubes_and_numbers(image):
    """Extract cubes and their numbers from an image

    Args:
        image: Either a PIL Image object, OpenCV image, or a path to an image file

    Returns:
        Tuple of (cube_count, extracted_numbers)
    """
    # Handle different input types
    if isinstance(image, str):
        # It's a file path
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # It's a PIL Image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Assume it's already an OpenCV image
        img = image

    # Use common preprocessing
    thresh = preprocess_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # We need both thresh and gray

    # Save debug image if needed
    from core.screencapture import get_debug_mode
    if get_debug_mode():
        try:
            # Ensure directory exists
            debug_dir = "./assets/screenshots"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)

            Image.fromarray(thresh).save("./assets/screenshots/cubes_thresholded.png")
        except Exception as e:
            logging.error(f"Error saving cubes thresholded image: {str(e)}")
            # Continue execution even if saving fails

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cubes = []
    for cnt in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter contours based on size (to exclude noise)
        if 20 < w < 120 and 20 < h < 120:  # Adjust based on cube size
            cubes.append((x, y, w, h))

    cubes = non_maximum_suppression(cubes)
    # Sort cubes based on their position (left-to-right, top-to-bottom)
    cubes = sorted(cubes, key=lambda c: (c[1], c[0]))

    # Extract numbers from each cube
    numbers = []
    i = 0
    for (x, y, w, h) in cubes:
        i += 1
        # Extract each cube
        cube = gray[y:y+h, x:x+w]

        # Resize the cube for better OCR accuracy
        cube_resized = cv2.resize(cube, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(cube_resized)
        img.save("./assets/screenshots/cubes/cube_{}.png".format(i))

        # Perform OCR
        text = pytesseract.image_to_string(cube_resized, config='--psm 6 -c tessedit_char_whitelist=0123456789')
        text = text.strip()

        if text.isdigit():
            numbers.append(int(text))

    return len(cubes), numbers

def extract_bet_size(image):
    """
    Extract the bet size from a specific region in the app.

    Args:
        image: Either a PIL Image object, OpenCV image, or a path to an image file

    Returns:
        bet_size: Detected bet size as an integer, or None if not found.
    """
    # Handle different input types
    if isinstance(image, str):
        # It's a file path
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # It's a PIL Image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Assume it's already an OpenCV image
        img = image

    # Use common preprocessing with CLAHE
    thresh = preprocess_image(img, use_clahe=True)

    # Save debug image if needed
    from core.screencapture import get_debug_mode
    if get_debug_mode():
        try:
            # Ensure directory exists
            debug_dir = "./assets/screenshots"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)

            Image.fromarray(thresh).save("./assets/screenshots/bet_thresholded.png")
        except Exception as e:
            logging.error(f"Error saving bet thresholded image: {str(e)}")
            # Continue execution even if saving fails

    # Resize the cube for better OCR accuracy
    bet_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    text = pytesseract.image_to_string(bet_resized, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
    text = text.strip()
    text = text.replace("$", "")  # Remove dollar sign if present
    logging.debug(f"Extracted bet size text: {text}")
    try:
        bet_size = int(float(text))  # Convert to integer if valid
        #eturn max(math.ceil(bet_size / 5 * 2), 1) * 10  # Adjust for the actual bet size because of DKK issues
        return bet_size * 10
    except ValueError:
        return None

def lobby_is_speed_baccarat(screenshot):
    """
    Determines if the lobby screen is for Speed Baccarat based on a template match.
    :param screenshot: Captured image of the lobby.
    :return: True if Speed Baccarat is detected, False otherwise.
    """
    logging.debug("Checking for Speed Baccarat lobby...")

    # Use noise removal and smoothing
    thresh = preprocess_image_v2(screenshot)

    # Resize the image for better OCR accuracy
    screenshot_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Save screenshot
    Image.fromarray(screenshot_resized).save("./assets/screenshots/is_speed_baccarat_thresholded.png")
    text = pytesseract.image_to_string(screenshot_resized, config="--psm 7 --oem 3")
    text = text.strip()
    #logging.info(f"Detected text: {text}")

    if text.upper().startswith("SPEED BACCARAT"):
        logging.debug("Speed Baccarat lobby detected.")
        return True
    else:
        logging.debug("Speed Baccarat lobby not detected.")
        return False