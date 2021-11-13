# Disaster Response Pipeline Project

### Overview

This project was made to analyze disaster data given for us by Figure Eight.

The main objective here is to improve the response for given necessities when a disaster occours, so, to accomplish this, I have built a Machine Learning model to identify potentially helpful messages.

The project is strustured as follows:

I. ETL Pipeline
	* Loads the message and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores the data in a SQLite database.
    
II. Machine Learning Pipeline
	* Loads the data from the SQLite database
    * Splits the data into training and testing datasets
    * Builds a text processing and Machine Learning pipeline
    * Trains and tunes the model using GridSearchCV (Cross Validation)
    * Outputs the results on the test set
    * Exports the final model as a pickle file
    
III. Web app
	* The web app was built upon the Flask application
    * I simply have modified the filepaths
    * Provides two simples visulizations about the train data.
    
### Data

The data used in this project were provided by Figure Eight, and they are:

	I. Disaster messages
    II. Disaster categories
    
### Resources

Some resources and tools used in this prtoject:

	* Python 3
    * Anaconda
    * Jupyter Notebooks
    * Pandas
    * Numpy
    * Scikit-learn
    * Re
    * SQLite3
    * Sqlalchemy
    * Pickle
    * NLTK

### Instructions:

1. Run the following commands in the project's root directory to set up the model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

### Author:

Guilherme Teixeira de Mello

	* GitHub profile: https://github.com/guilhermemello07
    * LinkedIn profile: https://www.linkedin.com/in/guilherme-demello/
    * Medium profile: https://guilhermemello1988.medium.com/
