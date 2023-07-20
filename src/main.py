import openai
import config
import transcribe
import embedding
import classification
from utils.input_utils import get_user_input


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
        print("Initiating transcription...")
        transcribe.transcribe(whisper_model, config.data_dir)
        print("Transcription done.")
    else:
        print("Transcription skipped.")

    embedding_prompt = get_user_input("Would you like the embeddings to be (re-)created? (yes/no): ", yes_choices
                                      + no_choices)

    if embedding_prompt in yes_choices:
        print("Initiating embedding...")
        df_text = embedding.text_to_csv(config.diagnosis_train_data)
        embedding.merge_embeddings_with_scores(df_text, config.diagnosis_train_scores)
        df_tokenization = embedding.tokenization(tokenizer)
        embedding.create_embeddings(df_tokenization)
        print("Embedding done.")
    else:
        print("Embedding skipped.")

    classification_prompt = get_user_input("Would you like the classification to be (re-)run? (yes/no): ", yes_choices
                                           + no_choices)

    if classification_prompt in yes_choices:
        print("Initiating classification...")
        df_embeddings_array = classification.embeddings_to_array()
        classification.classify_embedding(df_embeddings_array)
        print("Classification done.")
    else:
        print("Classification skipped.")


if __name__ == "__main__":
    main()
