from discord_webhook import DiscordWebhook
import json
import os

def load_creds():
    """Load credentials from JSON file"""
    creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'creds.json')
    with open(creds_path, 'r') as file:
        return json.load(file)

def send_discord_notification(message):
    creds = load_creds()
    webhook_url = creds.get('discord_webhook_url')
    if not webhook_url:
        print("Warning: Discord webhook URL not configured")
        return
        
    webhook = DiscordWebhook(url=webhook_url, content=message)
    webhook.execute()

def on_line_finish(profit: float, reason: str, is_second_shoe: bool = False):
    """Send a Discord message when a line finishes"""
    shoe_type = "Second shoe" if is_second_shoe else "First shoe"
    message = f"{shoe_type} finished with {profit}! Reason: {reason}"
    send_discord_notification(message)

def on_error(error):
    """
    This function should be called whenever an error occurs.
    It sends a message to Discord with the error message.
    """
    message = f"An error occurred: {error}"
    send_discord_notification(message)

def on_start_bot():
    """
    This function should be called when the bot starts.
    It sends a message to Discord.
    """
    message = "Bot started."
    send_discord_notification(message)

def on_table_joined():
    """
    This function should be called when the bot joins a table.
    It sends a message to Discord.
    """
    message = "Bot joined a table."
    send_discord_notification(message)