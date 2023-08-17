import logging

# Configure logging to display messages in the terminal
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this file
log = logging.getLogger("Input utils")

def get_user_input(prompt, choices):
    while True:
        user_input = input(prompt)
        if user_input.lower() in choices:
            return user_input.lower()
        else:
            log.info('Invalid input. Please try again.')