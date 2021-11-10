import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads two datasets and merges them into a pandas dataframe
    
    INPUT
    messages_filepath = the path for the messages file
    categories_filepath = the path for the categories file
    
    OUTPUT
    df = a pandas dataframe that has data from the previous datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='outer', on='id')
    
    return df


def clean_data(df):
    '''
    INPUT
    df = a pandas dataframe that has data for the messages and categories datasets
    
    OUTPUT
    df = a clean version of the dataframe
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use the row to extract a list of new column names
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set the value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from the df
    df.drop('categories', axis='columns', inplace=True)
    # concatenate the original df with the new categories df
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop all the rows where the value for the related column is 2
    df.drop(index=df.loc[df['related'] == 2].index.tolist(), inplace=True)
    
    return df


def save_data(df, database_filepath):
    '''
    INPUT
    df = the cleaned pandas dataframe
    database_filename = the name of the SQL database
    
    OUTPUT = the cleaned df loaded to the database_filename sql server
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()