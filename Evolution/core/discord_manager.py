from discord_webhook import DiscordWebhook

def send_discord_notification(message):
    webhook_url = "https://discord.com/api/webhooks/1340721807254356059/VYyHrhaXPqZqHvP88pCKCV5bRLaXkSSqzo8efah67XJYwZjFq2B7OxOnlg_4-YpQWA5a"
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