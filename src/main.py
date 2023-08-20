import openai

import classification
import config
import embedding
import transcribe
from config import logger
from utils.input_utils import get_user_input


def main():
    openai.api_key = config.secret_key()

    tokenizer = config.set_up()

    ### User input
    yes_choices = ["yes", "y"]
    no_choices = ["no", "n"]

    transcription_prompt = get_user_input("Would you like the transcriptions to be (re-)created? (yes/no): ",
                                          yes_choices
                                          + no_choices)

    if transcription_prompt in yes_choices:
        whisper_model_choices = ["tiny", "base", "small", "medium", "large"]
        whisper_model_prompt = get_user_input("Which Whisper model should be used for transcription? "
                                              "(tiny/base/small/medium/large): ", whisper_model_choices)
        transcribe.transcribe(whisper_model_prompt)
    else:
        logger.info("Transcription skipped.")

    classification_prompt = get_user_input("Would you like the classification to be (re-)run? (yes/no): ", yes_choices
                                           + no_choices)

    if classification_prompt in yes_choices:
        classification_type_choices = ["embedding", "acoustic"]
        classification_type_prompt = get_user_input("Which classification method would you like to use? "
                                                    "(embedding/acoustic): ", classification_type_choices)

        if classification_type_prompt == "embedding":
            embedding_prompt = get_user_input("Would you like the embeddings to be (re-)created? (yes/no): ",
                                              yes_choices
                                              + no_choices)
            if embedding_prompt in yes_choices:
                logger.info("Initiating embedding...")
                df_text = embedding.text_to_csv(config.diagnosis_train_data)
                embedding.merge_embeddings_with_scores(df_text, config.diagnosis_train_scores)
                df_tokenization = embedding.tokenization(tokenizer)
                embedding.create_embeddings(df_tokenization)
                logger.info("Embedding done.")
            else:
                logger.info("Embedding skipped.")

            df_embeddings_array = classification.embeddings_to_array()
            classification.classify_embedding(df_embeddings_array, config.n_splits)
        elif classification_type_prompt == "acoustic":
            classification.classify_acoustic()
    else:
        logger.info("Classification skipped.")


if __name__ == "__main__":
    main()
