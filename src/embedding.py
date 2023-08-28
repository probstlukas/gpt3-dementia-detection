from matplotlib import pyplot as plt
import config
from config import logger
import pandas as pd
import openai


def tokenization(df, tokenizer):
    """
    Tokenize text in a DataFrame and handle chunking for texts exceeding a maximum token limit.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing 'addressfname' and 'transcript' columns.
        tokenizer: The tokenizer object used to tokenize the text.

    Returns:
        pandas.DataFrame: Processed DataFrame with added 'n_tokens' column and potentially split text chunks.
    """
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df['transcript'].apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df['n_tokens'].hist()
    plt.show()

    def split_into_many(text, max_tokens):
        """
        Split a long text into multiple chunks, each with a maximum number of tokens.

        Parameters:
            text (str): The input text to be split into chunks.
            max_tokens (int): The maximum number of tokens allowed in each chunk.

        Returns:
            list: A list of text chunks, where each chunk has at most max_tokens tokens.
        """
        sentences = text.split('. ')

        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence in sentences:
            token = len(tokenizer.encode(" " + sentence))

            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            if token > max_tokens:
                continue

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    shortened = []

    for _, row in df.iterrows():
        if row['transcript'] is None:
            continue

        if row['n_tokens'] > config.max_tokens:
            row_chunks = split_into_many(row['transcript'], max_tokens=config.max_tokens)
            for chunk in row_chunks:
                shortened.append({'addressfname': row['addressfname'], 'transcript': chunk})
        else:
            shortened.append({'addressfname': row['addressfname'], 'transcript': row['transcript']})

    df_shortened = pd.DataFrame(shortened)
    df_shortened['n_tokens'] = df_shortened['transcript'].apply(lambda x: len(tokenizer.encode(x)))
    df_shortened['n_tokens'].hist()
    plt.show()

    logger.debug(df_shortened)
    return df_shortened


def create_embeddings(df):
    df['embedding'] = df['transcript'].apply(
        lambda x: openai.Embedding.create(input=x, engine=config.embedding_engine)['data'][0]['embedding'])
    df = df.drop('transcript', axis=1)
    return df


def embeddings_exists():
    if config.train_embeddings_path.is_file() and config.test_embeddings_path.is_file():
        return True
    return False
