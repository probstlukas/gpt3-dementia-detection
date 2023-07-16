import openai

import config
import transcribe
import embedding
import classification
from utils.input_utils import get_user_input


def main():
    openai.api_key = config.secret_key()

    tokenizer = config.set_up()

    ### User input
    yes_choices = ["yes", "y"]
    no_choices = ["no", "n"]

    transcription_prompt = get_user_input("Would you like the transcriptions to be recreated? (yes/no): ", yes_choices
                                          + no_choices)

    if transcription_prompt in yes_choices:
        transcribe.transcribe(config.model, config.data_dir)
    else:
        print("Transcription skipped.")

    embedding_prompt = get_user_input("Would you like the embeddings to be recreated? (yes/no): ", yes_choices
                                      + no_choices)

    if embedding_prompt in yes_choices:
        df_text = embedding.text_to_csv(config.diagnosis_train_data)
        embedding.merge_embeddings_with_scores(df_text, config.diagnosis_train_scores)
        df_tokenization = embedding.tokenization(tokenizer)
        embedding.create_embeddings(df_tokenization)
    else:
        print("Embedding skipped.")

    ### Classification
    print("Initiating classification...")
    df_embeddings_array = classification.embeddings_to_array()
    classification.classify_svc(df_embeddings_array)
    classification.classify_lr(df_embeddings_array)
    classification.classify_rf(df_embeddings_array)


if __name__ == "__main__":
    main()
