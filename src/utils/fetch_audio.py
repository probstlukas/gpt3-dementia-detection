import os
import logging

# Configure logging to display messages in the terminal
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this file
log = logging.getLogger("Fetch audio")


def fetch_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    log.info(f"Successfully fetched {len(audio_files)} (.wav) audio files!")
    return audio_files
