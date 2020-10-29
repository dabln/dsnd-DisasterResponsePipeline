# Web app for NLP classification
Part of the Udacity Data Science Nanodegree:
__Disaster Response Project__

## General info
A web app is created with Flask and Bootstrap for Natural Language Processing (__NLP__). The app provides an interface for new messages, e.g. Twitter messages scanned by disaster relief agencies in a _Disaster Response_ situation.

A new __message is classified__ into categories -- like 'aid related', 'search and rescue', 'child alone', and 'water' -- based on the learnings from the labeled training data which contains real messages that were sent during disaster events.

New training data can be provided and used to update the model. More precisely, data cleaning and storing in a database can be performed using an __ETL pipeline__ and training the classifier and providing the best model to the web app can be performed using a __Machine Learning (ML) pipeline__.

## Requirements
Python 3 mainly with the packages Pandas, flask, plotly, nltk and sqlalchemy to run the web app. To use the pipelines mainly numpy and sklearn
are needed. The code from the package [HyperclassifierSearch](https://github.com/dabln/HyperclassifierSearch) created by the same author is used as well.

### Instructions:
To run the web app:
- Execute the Python file 'run.py' in the 'app' folder via the command line:
    `python run.py`
- Go to http://0.0.0.0:3001/ in a browser.

To run the pipelines:
- Run the following commands in the project's root directory.
- Run the ETL pipeline via the command line:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- Run the ML pipeline via the command line:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
