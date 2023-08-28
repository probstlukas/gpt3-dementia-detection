from config import logger
from pathlib import Path
import config
from utils.utils import fetch_audio_files
import opensmile
import pandas as pd


def feature_vectors_exists():
    if config.acoustic_results_file.exists():
        return True
    else:
        return False


def save_feature_vectors(transcription_csv):
    file_label_dict = dict(zip(transcription_csv['addressfname'], transcription_csv['diagnosis']))

    # Create an instance of the openSMILE feature extractor
    smile_lowlevel = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        loglevel=2,
        logfile=Path(config.acoustic_results_dir / 'smile_lowlevel.log').resolve(),
        num_workers=5  # Speed up process by parallelizing
    )
    smile_functionals = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        loglevel=2,
        logfile=Path(config.acoustic_results_dir / 'smile_functionals.log').resolve(),
        num_workers=5  # Speed up process by parallelizing
    )

    # Fetch the audio files
    audio_files = fetch_audio_files(config.diagnosis_train_data)
    # Create lists to hold the filenames, labels, and feature vectors
    filenames = []
    labels = []
    feature_vectors_lowlevel = []
    feature_vectors_functionals = []
    # Iterate through audio files and extract features
    for audio_file in audio_files:
        filename = Path(audio_file).stem
        label = file_label_dict.get(filename)
        logger.debug(label)

        # Check if the label is not None (i.e., the filename was found in the DataFrame)
        if label is not None:
            filenames.append(filename)
            labels.append(label)

            # Extract Low-Level Descriptors using openSMILE
            feature_vector_lowlevel = smile_lowlevel.process_file(audio_file)
            logger.debug(feature_vector_lowlevel)
            feature_vectors_lowlevel.append(feature_vector_lowlevel)

            # Extract Functionals features using openSMILE
            feature_vector_functionals = smile_functionals.process_file(audio_file)
            logger.debug(feature_vector_functionals)
            feature_vectors_functionals.append(feature_vector_functionals)
    # Combine Low-Level Descriptors and Functionals features
    #feature_vectors_combined = [lowlevel + functionals for lowlevel, functionals in
                                #zip(feature_vectors_lowlevel, feature_vectors_functionals)]
    #logger.debug(feature_vectors_combined)

    # Create a DataFrame from the feature vectors
    data = pd.DataFrame(feature_vectors_functionals)
    # Write the DataFrame to the CSV file using pandas
    data.to_csv(config.acoustic_results_file, index=False)

    # Create a DataFrame from the feature vectors
    data = pd.DataFrame(feature_vectors_lowlevel)
    # Write the DataFrame to the CSV file using pandas
    #data.to_csv(Path(config.acoustic_results_dir /).resolve(), index=False)
    logger.info(f"Writing {config.acoustic_results_file}...")
