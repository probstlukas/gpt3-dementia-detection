import openai
import config
import transcribe
import embedding
import classification
from utils.input_utils import get_user_input
import logging

# Configure logging to display messages in the terminal
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this file
log = logging.getLogger("Main")


def main():
    openai.api_key = config.secret_key()

    whisper_model, tokenizer = config.set_up()

    ### User input
    yes_choices = ["yes", "y"]
    no_choices = ["no", "n"]

    transcription_prompt = get_user_input("Would you like the transcriptions to be (re-)created? (yes/no): ",
                                          yes_choices
                                          + no_choices)

    if transcription_prompt in yes_choices:
        log.info("Initiating transcription...")
        transcribe.transcribe(whisper_model, config.data_dir)
        log.info("Transcription done.")
    else:
        log.info("Transcription skipped.")

    embedding_prompt = get_user_input("Would you like the embeddings to be (re-)created? (yes/no): ", yes_choices
                                      + no_choices)

    if embedding_prompt in yes_choices:
        log.info("Initiating embedding...")
        df_text = embedding.text_to_csv(config.diagnosis_train_data)
        embedding.merge_embeddings_with_scores(df_text, config.diagnosis_train_scores)
        df_tokenization = embedding.tokenization(tokenizer)
        embedding.create_embeddings(df_tokenization)
        log.info("Embedding done.")
    else:
        log.info("Embedding skipped.")

    classification_prompt = get_user_input("Would you like the classification to be (re-)run? (yes/no): ", yes_choices
                                           + no_choices)

    if classification_prompt in yes_choices:
        log.info("Initiating classification...")
        df_embeddings_array = classification.embeddings_to_array()
        classification.classify_embedding(df_embeddings_array, 10)
        # classification.classify_acoustic()
        log.info("Classification done.")
    else:
        log.info("Classification skipped.")


if __name__ == "__main__":
    main()
