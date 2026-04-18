import os
import config
from config import logger
import pandas as pd


def fetch_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    logger.info(f"Successfully fetched {len(audio_files)} (.wav) audio files!")
    return audio_files


def get_user_input(prompt, choices):
    while True:
        user_input = input(prompt)
        if user_input.lower() in choices:
            return user_input.lower()
        else:
            logger.info('Invalid input. Please try again.')


def df_to_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        file_path (str): The path to the CSV file.

    Returns:
        None
    """
    # Save the DataFrame to a CSV file without including the index column
    df.to_csv(file_path, index=False)
    logger.info(f"Writing {file_path}...")


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
    df = df.copy()
    raw_diagnosis = (
        df['diagnosis']
        .astype(str)
        .str.strip()
        .str.lower()
    )
    diagnosis_map = {'ad': 1, 'cn': 0}
    df['diagnosis'] = raw_diagnosis.map(diagnosis_map)

    if df['diagnosis'].isna().any():
        invalid_labels = sorted(raw_diagnosis[df['diagnosis'].isna()].unique())
        raise ValueError(f"Unexpected diagnosis labels found: {invalid_labels}")

    return df
