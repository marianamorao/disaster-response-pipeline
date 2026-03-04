"""
Machine Learning pipeline for the Disaster Response Pipeline project.

Usage:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Arguments:
    database_filepath : Path to the SQLite database containing cleaned data.
    model_filepath    : Path where the trained model pickle file will be saved.
"""

import sys
import pickle
import re
import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters
    ----------
    database_filepath : str
        Path to the SQLite database file.

    Returns
    -------
    X : pd.Series
        Feature column (messages).
    Y : pd.DataFrame
        Target columns (36 categories).
    category_names : list of str
        Names of the 36 category columns.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Drop columns with only one unique value (LinearSVC requires at least 2 classes)
    Y = Y.loc[:, Y.nunique() > 1]

    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and clean a text string.

    Steps:
    - Normalize to lowercase and remove punctuation.
    - Split into words.
    - Remove common English stop words.

    Parameters
    ----------
    text : str
        Raw text string.

    Returns
    -------
    list of str
        List of cleaned tokens.
    """
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'not', 'no', 'so', 'as', 'if', 'than', 'then', 'when', 'where'
    }
    text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text).lower())
    tokens = text.split()
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV.

    The pipeline:
    1. Vectorizes text using CountVectorizer with the custom tokenize function.
    2. Applies TF-IDF transformation.
    3. Fits a MultiOutputClassifier with LinearSVC.

    LinearSVC is significantly faster and more accurate than RandomForest
    for text classification tasks.

    GridSearchCV searches over a small parameter grid to find the best model.

    Returns
    -------
    GridSearchCV
        The configured grid search object wrapping the pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=10000)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(random_state=42, max_iter=1000)))
    ])

    # Parameter grid for GridSearchCV (kept small for reasonable training time)
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__C': [0.1, 1.0],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model and print a classification report per category.

    Outputs precision, recall, and f1-score for each of the 36 categories.

    Parameters
    ----------
    model : GridSearchCV or Pipeline
        Trained model.
    X_test : pd.Series
        Test feature data.
    Y_test : pd.DataFrame
        True labels for the test set.
    category_names : list of str
        Names of the target categories.
    """
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print('-' * 60)


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Parameters
    ----------
    model : GridSearchCV or Pipeline
        Trained model object.
    model_filepath : str
        Destination path for the pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main function to run the ML pipeline.

    Loads data from the database, builds and trains a model,
    evaluates it on the test set, and saves it as a pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument and the filepath of the pickle file to '
            'save the model to as the second argument.\n\nExample: python '
            'train_classifier.py ../data/DisasterResponse.db classifier.pkl'
        )


if __name__ == '__main__':
    main()
