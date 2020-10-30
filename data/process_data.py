# ETL pipeline
# example use in comandline: python process_data.py disaster_messages.csv disaster_categories.csv disaster.db

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # read and merge csv's
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    if messages.id.isin(categories.id).sum() != messages.shape[0]:
        print('error: different IDs in the data sets')
        sys.exit()
    return pd.merge(messages, categories, on='id', how='inner')

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories[categories.index == 0]

    # extract a list of new column names for categories.
    category_colnames = pd.Series(row.values[0]).apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # category values to numbers 0 or 1
    for column in categories:
        categories[column] = pd.Series(categories[column].values).apply(lambda x: x[-1:])
        categories[column] = categories[column].astype('int64')

    if (categories.isna().sum().sum() > 0):
        print('error: issue with NaN values')
        sys.exit()

    # Replace categories column in df with new category columns.
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # removing invalid values for column 'related'
    df = df[df.related != 2]

    # drop duplicates
    df.drop_duplicates(inplace=True, subset='id')

    return df

def save_data(df, database_filename):
    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_table', engine, index=False, if_exists='replace')

    # check that DB was filled...
    print('    Check database content... table exists and has entries:',
      pd.read_sql('SELECT * FROM disaster_table',
                  con=engine).shape[0] > 0)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # assert inspired by https://www.tutorialspoint.com/python3/assertions_in_python.htm:
        assert(messages_filepath[-4:] == '.csv'), 'no .csv file path given'
        assert(categories_filepath[-4:] == '.csv'), 'no .csv file path given'
        assert(database_filepath[-3:] == '.db'), 'no .db file path given'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('    dataframe shape is', df.shape)

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
