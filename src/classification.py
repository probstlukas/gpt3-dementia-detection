import config
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer, recall_score, precision_score, f1_score

"""
Create embeddings of the control group, and compare it with the embeddings of the diagnosed group.
The text-davinci-003 model will then evaluate it.
"""


# Turning the embeddings into a NumPy array, which will provide more flexibility in how to use it.
# It will also flatten the dimension to 1-D, which is the required format for many subsequent operations.
def embeddings_to_array():
    df = pd.read_csv(config.embeddings_path, index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()
    return df


# Helper function for specific classify functions
def classify(df):
    # Drop rows with NaN values in "embedding" or "dx" columns
    # TODO: There shouldn't be any empty entries in the first place
    df.dropna(subset=['embeddings', 'dx'], inplace=True)

    #############################
    # Define the dependent variable that needs to be predicted (labels)
    y = df['dx'].values

    # Define the independent variable
    X = list(df['embeddings'].values)
    # Create models
    models = [SVC(), LogisticRegression(), RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Define custom scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='macro')
    }

    # Prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=42, shuffle=True)

    df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    for model, name in zip(models, names):
        # Perform cross-validation with custom scoring metrics
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        # print(name)
        # print(scores)

        train_accuracy_mean = round(scores['train_accuracy'].mean(), 3)
        train_accuracy_std = round(scores['train_accuracy'].std(), 3)
        train_precision_mean = round(scores['train_precision'].mean(), 3)
        train_precision_std = round(scores['train_precision'].std(), 3)
        train_recall_mean = round(scores['train_recall'].mean(), 3)
        train_recall_std = round(scores['train_recall'].std(), 3)
        train_f1_mean = round(scores['train_f1_score'].mean(), 3)
        train_f1_std = round(scores['train_f1_score'].std(), 3)

        df = pd.concat([df, pd.DataFrame([{'Set': 'Train',
                                           'Model': name,
                                           'Accuracy': f"{train_accuracy_mean} "
                                                       f"({train_accuracy_std})",
                                           'Precision': f"{train_precision_mean} "
                                                        f"({train_precision_std})",
                                           'Recall': f"{train_recall_mean} "
                                                     f"({train_recall_std})",
                                           'F1': f"{train_f1_mean} "
                                                 f"({train_f1_std})",
                                           }])], ignore_index=True)

        test_accuracy_mean = round(scores['test_accuracy'].mean(), 3)
        test_precision_mean = round(scores['test_precision'].mean(), 3)
        test_recall_mean = round(scores['test_recall'].mean(), 3)
        test_f1_mean = round(scores['test_f1_score'].mean(), 3)

        df = pd.concat([df, pd.DataFrame([{'Set': 'Test',
                                           'Model': name,
                                           'Accuracy': test_accuracy_mean,
                                           'Precision': test_precision_mean,
                                           'Recall': test_recall_mean,
                                           'F1': test_f1_mean
                                           }])], ignore_index=True)

    df = df.sort_values(by='Set', ascending=False)
    df = df.reset_index(drop=True)
    # print(df)

    df.to_csv(config.results_path)
    print(f"Writing {config.results_path}...")
