import openai
import pandas as pd

import classification
import config
import embedding
import transcribe
import acoustic_features
from config import logger
from utils.utils import get_user_input, df_to_csv


def main():
    openai.api_key = config.secret_key()

    tokenizer = config.set_up()

    ### User input
    yes_choices = ["yes", "y"]
    no_choices = ["no", "n"]

    transcription_prompt = get_user_input("Would you like to transcribe the audio files? (yes/no): ",
                                          yes_choices
                                          + no_choices)

    if transcription_prompt in yes_choices:
        logger.info("If there is already a transcription, please delete it first. "
                    "Otherwise, already transcribed files will be skipped, no matter which model was used for it.")
        whisper_model_choices = ["tiny", "base", "small", "medium", "large"]
        whisper_model_prompt = get_user_input("Which Whisper model should be used for transcription? "
                                              "(tiny/base/small/medium/large): ", whisper_model_choices)
        config.whisper_model_name = whisper_model_prompt
        transcribe.transcribe()
    else:
        logger.info("Transcription skipped.")

    classification_prompt = get_user_input("Would you like the classification to be (re-)run? (yes/no): ", yes_choices
                                           + no_choices)

    if classification_prompt in yes_choices:
        classification_type_choices = ["embedding", "acoustic"]
        classification_type_prompt = get_user_input("Which classification method would you like to use? "
                                                    "(embedding/acoustic): ", classification_type_choices)

        if classification_type_prompt == "embedding":
            create_embeddings = False
            # Check if there are already older embeddings
            if embedding.embeddings_exists():
                embedding_prompt = get_user_input("There already seem to exist some embeddings. "
                                                  "Would you like to create new embeddings? (yes/no): ",
                                                  yes_choices + no_choices)
                if embedding_prompt in yes_choices:
                    create_embeddings = True
                else:
                    logger.info("Embedding skipped.")
            else:
                create_embeddings = True
                logger.info("Embeddings not found. Creating embeddings automatically...")

            if create_embeddings:
                logger.info("Initiating embedding...")
                # Read transcriptions and prepare csv files
                train_df = pd.read_csv(config.train_scraped_path)
                test_df = pd.read_csv(config.test_scraped_path)

                # Tokenization
                train_tokenization = embedding.tokenization(train_df, tokenizer)
                test_tokenization = embedding.tokenization(test_df, tokenizer)

                # Create embeddings
                train_embeddings = embedding.create_embeddings(train_tokenization)
                test_embeddings = embedding.create_embeddings(test_tokenization)

                # Save embeddings to csv
                df_to_csv(train_embeddings, config.train_embeddings_path)  # Specify file paths
                df_to_csv(test_embeddings, config.test_embeddings_path)

                logger.info("Embedding done.")

            train_embeddings_array = classification.embeddings_to_array(config.train_embeddings_path)
            test_embeddings_array = classification.embeddings_to_array(config.test_embeddings_path)

            classification.classify_embedding(train_embeddings_array, test_embeddings_array, config.n_splits)

        elif classification_type_prompt == "acoustic":
            create_acoustic_features = False
            # Check if there are already older feature vectors
            if acoustic_features.feature_vectors_exists():
                acoustic_vectors_prompt = get_user_input("Acoustic feature vectors have been created in the past."
                                                         "Would you like to overwrite them? (yes/no): ",
                                                         yes_choices
                                                         + no_choices)
                if acoustic_vectors_prompt in yes_choices:
                    create_acoustic_features = True
                else:
                    logger.info("Acoustic feature vectors skipped.")
            else:
                create_acoustic_features = True
                logger.info("Acoustic feature vectors not found. Creating them automatically...")

            transcription_csv = pd.read_csv(config.train_scraped_path)
            if create_acoustic_features:
                logger.info("Initiating acoustic feature vectors...")
                acoustic_features.save_feature_vectors(transcription_csv)
                logger.info("Acoustic feature vectors done.")
            acoustic_features_csv = pd.read_csv(config.acoustic_results_file)
            classification.classify_acoustic(acoustic_features_csv, transcription_csv, config.n_splits)
    else:
        logger.info("Classification skipped.")


if __name__ == "__main__":
    main()
