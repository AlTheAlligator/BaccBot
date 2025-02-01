from core.interaction import click_button, random_mouse_move
from core.screencapture import capture_game_result, capture_bet_allowed, get_banker_bet_coordinates, get_player_bet_coordinates, get_10_bet_coordinates, get_50_bet_coordinates, get_250_bet_coordinates, get_500_bet_coordinates, get_1000_bet_coordinates, get_2000_bet_coordinates, get_bet_allowed_coordinates
import cv2
from PIL import Image
import pytesseract
import time
import random

def get_game_result():
    # Load the image
    results = capture_game_result()
    if results is None:
        return "Waiting for Results"
    img = cv2.imread(capture_game_result())

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY_INV, 11, 2)
    
    #thresh = cv2.equalizeHist(thresh)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_2 = Image.fromarray(th3)
    img_2.save("./assets/screenshots/result_thresholded.png")  # Save for debugging

    # Resize the cube for better OCR accuracy
    bet_resized = cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    text = pytesseract.image_to_string(bet_resized, config='--psm 6 -c tessedit_char_whitelist=BANKERPLYTIE')
    text = text.strip()
    
    if text == "BANKER":
        return "B"
 
    if text == "PLAYER":
        return "P"
    
    if text == "TIE":
        return "T"
        
    return "Waiting for Results"
    
def get_bet_allowed():
    # Load the image
    coordinates = get_bet_allowed_coordinates()
    if coordinates is None:
        return False
    return True
    
    img = cv2.imread(capture_bet_allowed())
    if img is None:
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    img_2 = Image.fromarray(thresh)
    img_2.save("./assets/screenshots/bet_allowed_thresholded.png")  # Save for debugging

    # Resize the cube for better OCR accuracy
    bet_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    text = pytesseract.image_to_string(bet_resized, config='--psm 10 -c tessedit_char_whitelist=50')
    text = text.strip()

    if text == "5000":
        return True
    
    return False

def wait_for_game_result():
    game_result = "Waiting for Results"

    while game_result == "Waiting for Results":
        game_result = get_game_result()
        random_mouse_move()

    return game_result
    
def wait_for_bet_allowed():
    bet_allowed = False

    while not bet_allowed:
        bet_allowed = get_bet_allowed()
        random_mouse_move()

    return bet_allowed

def place_bets(side, chips):
    last_chip = None
    for chip in chips:
        if chip != last_chip:
            if chip == 10:
                click_button(get_10_bet_coordinates())
            elif chip == 50:
                click_button(get_50_bet_coordinates())
            elif chip == 250:
                click_button(get_250_bet_coordinates())
            elif chip == 500:
                click_button(get_500_bet_coordinates())
            elif chip == 1000:
                click_button(get_1000_bet_coordinates())
            elif chip == 2000:
                click_button(get_2000_bet_coordinates())
            time.sleep(random.uniform(0.05, 0.1))  # Delay between chips
        
        if side == "P":
            click_button(get_player_bet_coordinates())
        elif side == "B":
            click_button(get_banker_bet_coordinates())

        last_chip = chip

        time.sleep(random.uniform(0.05, 0.1))  # Delay between bets