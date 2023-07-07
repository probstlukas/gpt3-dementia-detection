import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import config
# imports
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------------------------------------------------------------------------------------------------
# FINAL GOAL: create embeddings of the control group, and compare it with the embeddings of the diagnosed group.
# The text-davinci-003 model will then evaluate it.
# ----------------------------------------------------------------------------------------------------------------------


# Turning the embeddings into a NumPy array, which will provide more flexibility in how to use it.
# It will also flatten the dimension to 1-D, which is the required format for many subsequent operations.
def embeddings_to_array():
    df = pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()
    return df


def create_context(question, df, max_len):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        question,
        model=config.model,
        max_len=config.max_len,
        debug=False,
        max_tokens=150,
        stop_sequence=None):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


def classify(df):
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embeddings.values), df.mmse, test_size=0.2, random_state=42
    )

    # train random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    report = classification_report(y_test, preds)
    print(report)