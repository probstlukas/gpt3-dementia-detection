import config
from config import logger
from utils.fetch_audio import fetch_audio_files
from pathlib import Path
import whisper
import os
import codecs
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.utils import resample
from utils.df_conversion import df_to_csv


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
    train_df = transcription_to_df(config.diagnosis_train_transcription_dir)
    train_df = add_train_scores(train_df)

    test_df = transcription_to_df(config.diagnosis_test_transcription_dir)

    df_to_csv(train_df, config.train_scraped_path)
    df_to_csv(test_df, config.test_scraped_path)

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


def add_train_scores(df):
    # reading two csv files
    text_data = df
    logger.debug(text_data)
    scores_df = pd.read_csv(config.diagnosis_train_scores)
    # Rename columns for consistency
    scores_df = scores_df.rename(columns={'adressfname': 'addressfname', 'dx': 'diagnosis'})
    scores_df = binarize_labels(scores_df)
    logger.debug(scores_df)

    # using merge function by setting how='inner'
    output = pd.merge(text_data,
                      scores_df[['addressfname', 'mmse', 'diagnosis']],  # We don't want the key column here
                      on='addressfname',
                      how='inner')

    logger.debug(output)
    return output


def binarize_labels(df):
    # Transform into binary classification
    df['diagnosis'] = [1 if label == 'ad' else 0 for label in df['diagnosis']]
    # How many data points for each class?
    # print(df.dx.value_counts())
    # Understand the data
    # sns.countplot(x='dx', data=df)  # 1 - diagnosed   0 - control group

    ### Balance data by down-sampling majority class
    # Separate majority and minority classes
    df_majority = df[df['diagnosis'] == 1]  # 87 ad datapoints
    df_minority = df[df['diagnosis'] == 0]  # 79 cn datapoints
    # print(len(df_minority))
    # Undersample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    # Combine undersampled majority class with minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    # Display new class counts
    # print(df_downsampled.dx.value_counts())
    # sns.countplot(x='dx', data=df_downsampled)  # 1 - diagnosed   0 - control group
    plt.show()
    return df_downsampled
