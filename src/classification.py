import config
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

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

    # Transform into binary classification
    df['dx'] = [1 if b == 'ad' else 0 for b in df.dx]

    # How many data points for each class?
    print(df['dx'].value_counts())

    # Understand the data
    sns.countplot(x='dx', data=df)  # ad - diagnosed   cn - control group

    # Separate majority and minority classes
    df_majority = df[df.dx == 1]  # 86 ad datapoints
    df_minority = df[df.dx == 0]  # 76 cn datapoints

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=86,  # to match majority class
                                     random_state=42)  # reproducible results

    # Combine majority class with upsampled minority class
    df = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    df.dx.value_counts()
    # sns.countplot(x='dx', data=df)  # ad - diagnosed   cn - control group

    # Define the dependent variable that needs to be predicted (labels)
    y = df['dx'].values

    # Define the independent variable
    X = list(df['embeddings'].values)
    # Create models
    models = [SVC(kernel='linear', random_state=42),
              LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
              RandomForestClassifier(random_state=42, n_jobs=-1)]
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
    print("Beginning to train the models...")
    for model, name in zip(models, names):
        # Define the pipeline to include scaling and the model
        estimators = [('scaler', MinMaxScaler()), ('model', model)]
        pipeline = Pipeline(estimators)

        # Perform cross-validation with custom scoring metrics
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
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

    print("Training done.")
    df = df.sort_values(by='Set', ascending=False)
    df = df.reset_index(drop=True)
    # print(df)

    df.to_csv(config.results_path)
    print(f"Writing {config.results_path}...")
    plt.show()
