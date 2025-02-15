from discord_webhook import DiscordWebhook

def send_discord_notification(message):
    webhook_url = "https://discord.com/api/webhooks/1336062591239323818/YMnh0ivTI5xUYPwkGU5UH6RHyadjV7nGvR3ARsIhjOUGTzS3Y6i8_aFtRCSXjrGFNwyy"
    webhook = DiscordWebhook(url=webhook_url, content=message)
    webhook.execute()

def on_line_finish(result, reason):
    """
    This function should be called whenever a line finishes.
    It sends a message to Discord with the result.
    """
    message = f"Line finished with result: {result}, because of: {reason}"
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