import os

from matplotlib import pyplot as plt

import config
import pandas as pd
from IPython.display import display
import openai


# Saving the raw text into a CSV file
def text_to_csv(data_dir):
    texts = []
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) != 'transcription':
            continue
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                texts.append((file, text))
    print(texts)

    # Blank empty lines can clutter the text files and make them harder to process.
    # This function removes those lines and tidies up the files.
    def remove_newlines(serie):
        serie = serie.str.replace('\n', ' ')
        serie = serie.str.replace('\\n', ' ')
        serie = serie.str.replace('  ', ' ')
        serie = serie.str.replace('  ', ' ')
        return serie

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['adressfname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.adressfname + ". " + remove_newlines(df.text)
    return df

def merge_embeddings_with_scores(df, diagnosis_train_scores_file):
    # reading two csv files
    data1 = df
    data2 = pd.read_csv(diagnosis_train_scores_file)

    # using merge function by setting how='inner'
    output1 = pd.merge(data1,
                       data2[['adressfname', 'mmse', 'dx']], # We don't want the key column here
                       on='adressfname',
                       how='inner')

    # displaying result
    print(output1)
    output1.to_csv('processed/scraped.csv')
    output1.head()


### Tokenization

# The API has a limit on the maximum number of input tokens for embeddings.
# To stay below the limit, the text in the CSV file needs to be broken down into multiple rows.
# The existing length of each row will be recorded first to identify which rows need to be split.
def tokenization(tokenizer):
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['adressfname', 'text', 'mmse', 'dx']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()

    display(df)
    plt.show()

    max_tokens = 500

    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens=max_tokens, title=None):
        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append({"adressfname": title, "text": ". ".join(chunk) + "."})
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    shortened = []

    # Loop through the dataframe
    for _, row in df.iterrows():

        # If the text is None, go to the next row
        if row['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row['n_tokens'] > max_tokens:
            chunks = split_into_many(row['text'], title=row['adressfname'])
            shortened += chunks

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append({"adressfname": row['adressfname'],
                              "text": row['text'],
                              "mmse": row['mmse'],
                              "dx": row['dx']})

    df_shortened = pd.DataFrame(shortened)
    df_shortened['n_tokens'] = df_shortened.text.apply(lambda x: len(tokenizer.encode(x)))
    df_shortened.n_tokens.hist()
    plt.show()
    return df_shortened


def create_embeddings(df):
    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    df.to_csv('processed/embeddings.csv')
    df.head()


