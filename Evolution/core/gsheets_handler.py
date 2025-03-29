from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint as pp

from core.strategy import get_first_6_non_ties


def initialize_gsheets():
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("./assets/creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Al the Alligator Updated Pro Mode Tracking Sheet").get_worksheet_by_id(0)
    return sheet

def write_result_line(context):
    sheet = initialize_gsheets()
    cell = sheet.findall("", in_column=4)
    
    # Time played (minutes)
    #sheet.update_cell(cell[1].row, cell[1].col, round((datetime.now() - context.table.line_start_time).total_seconds() / 60))
    # Total units gained
    if context.get_total_pnl() <= -4000:
        sheet.update_cell(cell[1].row, cell[1].col + 1, abs(context.get_total_pnl()))
    else:
        sheet.update_cell(cell[1].row, cell[1].col, context.get_total_pnl())
    # PPP/BBB
    sheet.update_cell(cell[1].row, cell[1].col + 2, context.game.initial_mode)
    # First 6
    if context.game.is_second_shoe:
        first_6 = get_first_6_non_ties(context.game.first_shoe_outcomes)
    else:
        first_6 = get_first_6_non_ties(context.game.outcomes)
    sheet.update_cell(cell[1].row, cell[1].col + 4, ''.join(first_6))
    # Second shoe?
    if context.game.is_second_shoe:
        sheet.update_cell(cell[1].row, cell[1].col + 5, "Yes")
    else:
        sheet.update_cell(cell[1].row, cell[1].col + 5, "No")
        
    # Cubes left going into second shoe
    if context.game.end_line_reason == "Shoe finished" and context.game.cube_count > 0:
        sheet.update_cell(cell[1].row, cell[1].col + 6, context.game.first_shoe_drawdown)
        sheet.update_cell(cell[1].row, cell[1].col + 7, context.game.cube_count)
    # Second shoe first 6
    #if context.game.is_second_shoe:
    #    first_6 = get_first_6_non_ties(context.game.outcomes)
    #    sheet.update_cell(cell[1].row, cell[1].col + 9, ''.join(first_6))