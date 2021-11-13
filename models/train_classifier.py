import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


def load_data(database_filepath):
    '''
    INPUT
    database_filepath = the database filepath
    
    OUTPUT
    X and Y = dataframes containing the data we are interested
    category_names = names for each of the 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df.message.values
    Y = df.drop(columns = ['id','message','original','genre'])
    
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT
    text = a string
    
    OUTPUT
    clean_tokens = a list of processed and cleanned tokens
    '''
    re_pattern = r"[^a-zA-Z0-9]"
    
    text = text.lower()
    
    text = re.sub(re_pattern, ' ', text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token_lem = lemmatizer.lemmatize(token).strip()
        clean_token_verb = lemmatizer.lemmatize(clean_token_lem, pos='v')
        clean_tokens.append(clean_token_verb)
        
    return clean_tokens
        


def build_model():
    '''
    This function builds a ML model
    
    First, it created a Pipeline, then choose a parameter to tune, and finally,
    the function uses GridSearchCV to Cross-Validate the parameter.
    
    INPUT
    This function does not need any input
    
    OUTPUT
    cv = The Machine Learning model
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': (5,20)
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters)
        
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model = a ML model
    X_test = the x_test part of the dataset
    Y_test = the y_test part of the dataset
    category_names = the names of the 36 categories that we need to consider
    
    OUTPUT
    Although the function hasn't a return method, it prints out the
    classification report for each of the categories
    '''
    Y_pred = model.predict(X_test)
    
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
        
    for column in Y_test.columns:
        print('Report for category {}:'.format(column))
        print(classification_report(Y_test[column], Y_pred_df[column]), '\n\n')


def save_model(model, model_filepath):
    '''
    INPUT
    model = the ML model to save
    model_filepath = the path you want your model to be saved
    
    OUTPUT
    Just saves the model
    '''
    filename = model
    pickle.dump(filename, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
