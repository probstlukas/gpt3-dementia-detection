import os
import config
from config import logger
import pandas as pd
from sklearn.utils import resample
from matplotlib import pyplot as plt


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
