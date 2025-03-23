
from core.interaction import human_like_mouse_move, click_button, scroll_lobby
from core.ocr import extract_cubes_and_numbers
from core.screencapture import capture_nameless_cubes, get_games_button_coordinates, get_baccarat_button_coordinates, get_close_running_game_coordinates
from core.nameless import is_line_done
import time, random
from main import setup_logging

from core.gsheets_handler import get_sheet_data

#cube_count, extracted_numbers = extract_cubes_and_numbers(capture_nameless_cubes())

#print(cube_count)
#print(extracted_numbers)

#coordinates = get_close_running_game_coordinates()
#click_button(coordinates)
#time.sleep(0.2)

#scroll = random.randint(3, 4)
#for i in range(scroll):
#    scroll_lobby("down", 120)
#    time.sleep(0.1)

#time.sleep(5)

#scroll = random.randint(8, 10)
#for i in range(scroll):
#    scroll_lobby("down", 120)
#    time.sleep(0.1)

#from core.state_machine.lobby_states import AnalyzeLobbyState

#setup_logging()

#time.sleep(3)

#state = AnalyzeLobbyState(None)
#print(state.find_lobby_bias())

get_sheet_data()