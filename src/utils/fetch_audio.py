import os
from config import logger


def fetch_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    logger.info(f"Successfully fetched {len(audio_files)} (.wav) audio files!")
    return audio_files
