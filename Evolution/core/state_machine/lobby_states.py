import logging
import time
import random
import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image
from core.screencapture import get_games_button_coordinates, get_baccarat_button_coordinates
from core.interaction import click_button, scroll_lobby
from core.discord_manager import on_table_joined
from core.state_machine.state_machine_base import State
from core.state_machine.lobby_utils import find_lobby_bias, find_tables, extract_table_histories, evaluate_table_histories

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = f'C:\Program Files\Tesseract-OCR\\tesseract.exe'

class LobbyState(State):
    def __init__(self, name: str, context):
        super().__init__(name, context)

    def execute(self):
        """Execute the lobby state's logic"""
        logging.info("State: Accessing the lobby")
        if not self.context.game.is_second_shoe:
            self.context.reset_table()
        
        # Click lobby buttons
        click_button(get_games_button_coordinates())
        time.sleep(random.uniform(0.2, 0.4))
        click_button(get_baccarat_button_coordinates())
        time.sleep(random.uniform(0.5, 1))
        
        # Find and analyze tables
        self.context.game.bias = find_lobby_bias()
        if self.context.game.is_second_shoe:
            if self.context.game.bias == "P":
                logging.info("Exiting: Bias is Player in second shoe.")
                exit()
            else:
                self.context.game.bias = "B"
        table = None
        while table is None:
            table = self.find_suitable_table(self.context.game.bias)
            if table is None:
                time.sleep(random.uniform(8, 15))
        
        if table:
            self.context.table.coordinates = table
            click_button(table)
            on_table_joined()
            time.sleep(1)
            return "initial_analysis"
        
        return None  # Stay in lobby if no table found

    def find_suitable_table(self, bias):
        """Find a suitable table based on the bias"""

        # Second pass if no suitable table found
        scroll = random.randint(2, 3)
        for i in range(scroll):
            scroll_lobby("down")
            time.sleep(random.uniform(0.11, 0.2))

        # First pass at finding tables
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)[:, :, ::-1]
        table_boxes = find_tables(screenshot)
        filtered_boxes = self.remove_cooldown_tables(table_boxes)
        table_histories = extract_table_histories(screenshot, filtered_boxes, True)
        
        # Evaluate tables
        table = evaluate_table_histories(table_histories, bias)
        if table is not None:
            return table

        # Second pass if no suitable table found
        scroll = random.randint(8, 10)
        for i in range(scroll):
            scroll_lobby("down")
            time.sleep(random.uniform(0.11, 0.2))

        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)[:, :, ::-1]
        table_boxes = find_tables(screenshot)
        filtered_boxes = self.remove_cooldown_tables(table_boxes)
        table_histories = extract_table_histories(screenshot, filtered_boxes, True)
        
        table = evaluate_table_histories(table_histories, bias)
        if table is not None:
            return table

        # Return to top
        for i in range(20):
            scroll_lobby("up")
            time.sleep(random.uniform(0.05, 0.11))

        logging.warning("No suitable table found.")
        return None

    def remove_cooldown_tables(self, tables, cooldown_minutes=5):
        """Remove tables that are in cooldown period"""
        current_time = time.time()
        updated_cooldown = []
        
        # Filter out expired cooldowns
        for coords, timestamp in self.context.table.cooldown:
            if current_time - timestamp <= cooldown_minutes * 60:
                updated_cooldown.append((coords, timestamp))
            else:
                logging.info(f"Table {coords} removed from cooldown list.")
        
        self.context.table.cooldown = updated_cooldown
        
        # Filter out any cooled-down tables from 'tables' list
        return [t for t in tables if t not in [cd[0] for cd in updated_cooldown]]