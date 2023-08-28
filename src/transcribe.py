import config
from config import logger
from utils.fetch_audio import fetch_audio_files
from pathlib import Path
import whisper
import os
import codecs
from utils.df_conversion import df_to_csv
import pandas as pd


def transcribe():
    whisper_model = whisper.load_model(config.whisper_model_name)

    logger.info("Initiating transcription...")

    # Get a list of all the audio files in the data folder
    diagnosis_train_audio_files = fetch_audio_files(config.diagnosis_train_data)
    logger.debug(diagnosis_train_audio_files)
    diagnosis_test_audio_files = fetch_audio_files(config.diagnosis_test_data)
    logger.debug(diagnosis_test_audio_files)

    # Write transcriptions files
    write_transcription(diagnosis_train_audio_files, config.diagnosis_train_transcription_dir, whisper_model)
    write_transcription(diagnosis_test_audio_files, config.diagnosis_test_transcription_dir, whisper_model)

    # Scrape all transcriptions and save it to a csv file
    scrape_transcriptions(config.diagnosis_train_transcription_dir, config.train_scraped_path)
    scrape_transcriptions(config.diagnosis_test_transcription_dir, config.test_scraped_path)

    logger.info("Transcription done.")


def write_transcription(audio_files, transcription_dir, whisper_model):
    # Loop over all the audio files in the folder
    for audio_file in audio_files:
        # Get base filename
        filename = Path(audio_file).stem
        transcription_file = (transcription_dir / filename).resolve()

        # Do not transcribe again if the transcription exists already
        if not transcription_file.exists():
            result = whisper_model.transcribe(audio_file, fp16=False)
            transcription_str = str(result["text"])

            # Create subdirs if not existent
            transcription_file.parent.mkdir(parents=True, exist_ok=True)

            transcription_file.write_text(transcription_str)
            logger.info(f"Transcribed {transcription_file}...")


def transcription_to_df(data_dir):
    """
    Transforms transcriptions from text files into a DataFrame.

    Parameters:
        data_dir (str): The directory containing transcription files.

    Returns:
        pd.DataFrame: A DataFrame with columns 'addressfname' and 'transcript'.
    """
    texts = []

    # Traverse through the directory to fetch transcription files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Read the content of the file
            with codecs.open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                texts.append((file, text))

    # Create a DataFrame from the list of texts
    df = pd.DataFrame(texts, columns=['addressfname', 'transcript'])

    # Clean up the transcript column by removing newlines and extra spaces
    df['transcript'] = df['transcript'].str.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' ')

    # Sort the DataFrame by the 'addressfname' column in ascending order
    df = df.sort_values(by='addressfname')

    # Reset the index
    df = df.reset_index(drop=True)

    # Debugging: Print the resulting DataFrame
    logger.debug(df)

    return df


def scrape_transcriptions(data_dir, file_path):
    df = transcription_to_df(data_dir)
    df.to_csv(file_path)
    logger.info(f"Writing {file_path}...")