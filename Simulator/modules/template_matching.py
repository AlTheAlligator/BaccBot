import cv2
import numpy as np
from PIL import Image

result_templates = {
    "B": "Screenshots/banker.png",
    "P": "Screenshots/player.png",
    "T": "Screenshots/tie.png",
    "N": "Screenshots/blank.png"
}

def crop_history_box(image_path, crop_coordinates, save_path="Screenshots/cropped_history_box.png"):
    img = Image.open(image_path)
    cropped_img = img.crop(crop_coordinates)
    cropped_img.save(save_path)
    return cropped_img

def template_match(cropped_img, templates, grid_size=(6, 14)):
    img_array = np.array(cropped_img)

    cell_width = img_array.shape[1] // grid_size[1]
    cell_height = img_array.shape[0] // grid_size[0]

    outcomes = []
    for col in range(grid_size[1]):
        for row in range(grid_size[0]):
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cell = img_array[y1:y2, x1:x2]

            img = Image.fromarray(cell)
            img.save("./assets/screenshots/cells/cell_{}_{}.png".format(row, col))
            
            detected = "Unknown"
            for outcome, template_path in templates.items():
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is None:
                    continue
                result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > 0.8:
                    detected = outcome
                    break
            outcomes.append(detected)
    return outcomes

def locate_button(game_window_image, button_template_path, threshold=0.8):
    game_window = cv2.imread(game_window_image)
    button_template = cv2.imread(button_template_path)

    result = cv2.matchTemplate(game_window, button_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        print(f"Button template found at: {button_template_path}")
        return max_loc
    return None
