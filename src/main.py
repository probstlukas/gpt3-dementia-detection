import config
import transcribe
import embedding
import classification


def main():
    tokenizer, ad_data_folder, cn_data_folder, results_folder = config.set_up()

    ### Transcription
    # ...

    ### Embedding
    embedding.text_to_csv(ad_data_folder)
    df_tokenization = embedding.tokenization(tokenizer)
    embedding.create_embeddings(df_tokenization)

    ### Classification
    df_embeddings_array = classification.embeddings_to_array()

    ### Example questions
    classification.answer_question(df_embeddings_array, question="What is our newest embeddings model?")
    classification.answer_question(df_embeddings_array, question="What is ChatGPT?")


if __name__ == "__main__":
    main()