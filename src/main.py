import openai

import config
import transcribe
import embedding
import classification


def main():
    # TODO: Create CLI to query if the user wants to transcribe, embed, and classify.
    transcribe_on = False
    embedding_on = False

    openai.api_key = config.secret_key()

    tokenizer = config.set_up()

    ### Transcription
    if transcribe_on:
        transcribe.transcribe(config.model, config.data_dir)

    ### Embedding
    if embedding_on:
        df_text = embedding.text_to_csv(config.diagnosis_train_data)
        embedding.merge_embeddings_with_scores(df_text, config.diagnosis_train_scores)
        df_tokenization = embedding.tokenization(tokenizer)
        embedding.create_embeddings(df_tokenization)

    ### Classification
    df_embeddings_array = classification.embeddings_to_array()
    print(df_embeddings_array)
    classification.classify(df_embeddings_array)

    # ## Example questions print(classification.answer_question(df_embeddings_array, question="Is there any sentence
    # which is grammatically incorrect?"))


if __name__ == "__main__":
    main()
