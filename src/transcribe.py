import config
from utils.fetch_audio import fetch_audio_files
import logging
from pathlib import Path

# Configure logging to display messages in the terminal
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this file
log = logging.getLogger("Transcribe")


def transcribe(model):
    # Get a list of all the audio files in the data folder
    audio_files = fetch_audio_files(config.diagnosis_train_data)
    log.debug(audio_files)

    transcription_dir = (config.transcription_path / config.whisper_model_name).resolve()

    # Loop over all the audio files in the folder
    for audio_file in audio_files:
        # Get base filename
        filename = Path(audio_file).stem
        transcription_file = (transcription_dir / filename).resolve()

        # Do not transcribe again if the transcription exists already
        if not transcription_file.exists():
            result = model.transcribe(audio_file, fp16=False)
            transcription_str = str(result["text"])

            # Create subdirs if not existent
            transcription_file.parent.mkdir(parents=True, exist_ok=True)

            transcription_file.write_text(transcription_str)
            log.info(f"Transcribed {transcription_file}...")
