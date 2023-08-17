import os
import config
from utils.fetch_audio import fetch_audio_files


def transcribe(model, data_dir):
    # Get a list of all the audio files in the data folder
    audio_files = fetch_audio_files(config.diagnosis_train_data)

    # Loop over all the audio files in the folder
    for audio_file_path in audio_files:

        transcription_dir = os.path.join(os.path.dirname(audio_file_path), 'transcription')
        filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        transcription_file = os.path.join(transcription_dir, filename)

        # Check if the respective transcription folder exists, otherwise create it
        if not os.path.exists(transcription_dir):
            os.makedirs(transcription_dir)
        # Check if file was already transcribed, then skip. Note: if you want to use a different model, this should be
        # removed!
        else:
            if os.path.isfile(transcription_file):
                log.info(f"Skipped {transcription_file}, because it's already transcribed.")
            continue

        result = model.transcribe(audio_file_path, fp16=False)
        transcription = str(result["text"])

        with open(transcription_file, 'w') as f:
            f.write(transcription)
            log.info(f"Transcribed {transcription_file}...")
