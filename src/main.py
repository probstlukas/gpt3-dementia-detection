import openai

import config
import embedding
import classification


def main():
    embedding_on = False

    openai.api_key = config.secret_key()

    tokenizer, ad_data_folder, cn_data_folder, results_folder, diagnosis_train_scores_file = config.set_up()

    ### Transcription
    # ...

    ### Embedding
    if embedding_on:
        df_text = embedding.text_to_csv(ad_data_folder)
        embedding.merge_embeddings_with_scores(df_text, diagnosis_train_scores_file)
        df_tokenization = embedding.tokenization(tokenizer)
        embedding.create_embeddings(df_tokenization)

    ### Classification
    df_embeddings_array = classification.embeddings_to_array()
    print(df_embeddings_array)
    classification.classify(df_embeddings_array)

    ### Example questions
    #print(classification.answer_question(df_embeddings_array, question="Is there any sentence which is grammatically incorrect?"))


if __name__ == "__main__":
    main()