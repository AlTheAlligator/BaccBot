import cv2
import pytesseract
import numpy as np
from PIL import Image

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = f'C:\Program Files\Tesseract-OCR\\tesseract.exe'  # Update with your Tesseract path

def extract_cubes_and_numbers(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    img_3 = Image.fromarray(thresh)
    img_3.save("./assets/screenshots/cubes_thresholded.png")  # Save for debugging

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cubes = []
    for cnt in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter contours based on size (to exclude noise)
        if 5 < w and 5 < h:  # Adjust based on cube size
            cubes.append((x, y, w, h))

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

def extract_bet_size(image_path):
    """
    Extract the bet size from a specific region in the app.

    Args:
    - image_path: Path to the screenshot of the app.

    Returns:
    - bet_size: Detected bet size as an integer, or None if not found.
    """
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )

    #thresh = cv2.medianBlur(thresh, 3)
    
    img_3 = Image.fromarray(thresh)
    img_3.save("./assets/screenshots/bet_thresholded.png")  # Save for debugging

    # Resize the cube for better OCR accuracy
    bet_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    text = pytesseract.image_to_string(bet_resized, config='--psm 7 -c tessedit_char_whitelist=0123456789$')
    text = text.strip()
    text = text.replace("$", "")  # Remove dollar sign if present
    try:
        bet_size = int(float(text))  # Convert to integer if valid
        return bet_size * 10  # Adjust for the actual bet size because of DKK issues
    except ValueError:
        return None
