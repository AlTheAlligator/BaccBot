from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint as pp
import logging

from core.strategy import get_first_6_non_ties


def initialize_gsheets(second_shoe=False):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("./assets/creds.json", scope)
    client = gspread.authorize(creds)
    
    if second_shoe:
        # Access the dedicated sheet for Second Shoe Lines
        # Note: The actual format for a Google Sheet ID is a long alphanumeric string
        # If 1040988089 is a placeholder ID, replace with the real sheet ID in production
        sheet = client.open("Al The Alligator Pro Mode Tracking Sheet (Test)").get_worksheet_by_id(1040988089)
        logging.info("Accessed Second Shoe Lines worksheet for tracking")
    else:
        sheet = client.open("Al The Alligator Pro Mode Tracking Sheet (Test)").get_worksheet_by_id(0)
    
    return sheet

def write_result_line(context):
    # Determine if this is a second shoe line
    is_second_shoe = context.game.is_second_shoe
    
    # Initialize the appropriate Google Sheet
    sheet = initialize_gsheets(second_shoe=is_second_shoe)
    
    if is_second_shoe:
        # Write to the dedicated Second Shoe Lines sheet
        write_second_shoe(sheet, context)
    else:
        # Write to the regular lines sheet
        write_first_shoe(sheet, context)
        
def write_first_shoe(sheet, context):
    # Original code for regular lines
    cell = sheet.findall("", in_column=4)
    
    # Total units gained
    if context.get_total_pnl() <= -4000:
        sheet.update_cell(cell[1].row, cell[1].col + 1, context.get_total_pnl())
    else:
        sheet.update_cell(cell[1].row, cell[1].col, context.get_total_pnl())
    # PPP/BBB
    sheet.update_cell(cell[1].row, cell[1].col + 2, context.game.initial_mode)
    # First 6
    first_6 = get_first_6_non_ties(context.game.outcomes)
    sheet.update_cell(cell[1].row, cell[1].col + 4, ''.join(first_6))

    sheet.update_cell(cell[1].row, cell[1].col + 5, "No")
        
    # Cubes left going into second shoe
    if context.game.end_line_reason == "Shoe finished" and context.game.cube_count > 0:
        sheet.update_cell(cell[1].row, cell[1].col, 0)
        sheet.update_cell(cell[1].row, cell[1].col + 5, "Yes")
        sheet.update_cell(cell[1].row, cell[1].col + 6, abs(context.game.first_shoe_drawdown))
        sheet.update_cell(cell[1].row, cell[1].col + 7, context.game.cube_count)

def write_second_shoe(sheet, context):
    # Write to the dedicated Second Shoe Lines sheet
    cell = sheet.findall("", in_column=4)
    
    # Total units gained
    if context.get_total_pnl() <= -4000:
        sheet.update_cell(cell[1].row, cell[1].col + 1, context.get_total_pnl())
    else:
        sheet.update_cell(cell[1].row, cell[1].col, context.get_total_pnl())

    # First 6
    first_6 = get_first_6_non_ties(context.game.outcomes)
    sheet.update_cell(cell[1].row, cell[1].col + 3, ''.join(first_6))
    sheet.update_cell(cell[1].row, cell[1].col + 4, ''.join(first_6))

    sheet.update_cell(cell[1].row, cell[1].col + 5, "Completed")
