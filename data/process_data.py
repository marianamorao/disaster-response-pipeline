"""
ETL pipeline for the Disaster Response Pipeline project.

Usage:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Arguments:
    messages_filepath   : Path to the CSV file containing messages.
    categories_filepath : Path to the CSV file containing categories.
    database_filepath   : Path (including filename) for the output SQLite database.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets.

    Parameters
    ----------
    messages_filepath : str
        Path to the messages CSV file.
    categories_filepath : str
        Path to the categories CSV file.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataframe.

    Steps:
    - Split the 'categories' column into 36 individual category columns.
    - Rename columns using the category names extracted from the data.
    - Convert category values to binary (0 or 1).
    - Replace any value of 2 in 'related' column with 1.
    - Drop the original 'categories' column and remove duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for storage.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Use first row to extract column names
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        # Replace values > 1 with 1 (e.g. 'related' has value 2 in some rows)
        categories[column] = categories[column].clip(upper=1)

    # Drop original categories column and concatenate cleaned columns
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath):
    """
    Save the cleaned dataframe to a SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    database_filepath : str
        File path for the SQLite database (e.g. 'DisasterResponse.db').
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """
    Main function to run the ETL pipeline.

    Reads command-line arguments for file paths, loads, cleans,
    and stores the data in a SQLite database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print(
            'Please provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively, as '
            'well as the filepath of the database to save the cleaned data '
            'to as the third argument.\n\nExample: python process_data.py '
            'disaster_messages.csv disaster_categories.csv DisasterResponse.db'
        )


if __name__ == '__main__':
    main()
