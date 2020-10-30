# Web app for NLP classification
Part of the Udacity Data Science Nanodegree, the __Disaster Response Project__.

## General info
A __web app__ is created with Flask and Bootstrap for Natural Language Processing (__NLP__). The app provides an __interface__ for new messages, e.g. Twitter messages scanned by disaster relief agencies in a _Disaster Response_ situation.

A new __message is classified__ into categories – like 'aid related', 'search and rescue', 'child alone', and 'water' – based on the learnings from the labeled training data which contains real messages that were sent during disaster events.

__New training data__ can be provided and used to update the model. More precisely, data cleaning and storing in a database can be performed using an __ETL pipeline__, and training the classifier and providing the best model to the web app can be performed using a __Machine Learning (ML) pipeline__.

## Requirements
Python 3 mainly with the packages Pandas, flask, plotly, nltk and sqlalchemy to run the web app. To use the pipelines mainly numpy and sklearn
are needed. The code from the package [HyperclassifierSearch](https://github.com/dabln/HyperclassifierSearch) created by the same author is used as well.

## Instructions
To run the web app:
- Execute the Python file 'run.py' in the 'app' folder via the command line:
    `python run.py`
- Go to http://0.0.0.0:3001/ in a browser.

To run the pipelines:
- Run the ETL pipeline via the command line in the `data` folder:<br>
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
- Run the ML pipeline via the command line in the `models` folder. The best model and its score are printed:<br>
        `python train_classifier.py ../data/DisasterResponse.db disaster_model.pkl`

## Files
Folder `notebooks_code_development` contains Jupyter notebooks used to develop the pipelines.

`data` contains the ETL pipeline (`process_data.py`) and the CSV input files plus the ETL pipeline output, an SQLite database.

`models` contains the ML pipeline (`train_classifier.py`) with its output, i.e. a Python pickle file with the best model from testing different classifiers and parameters. That pickle file is used in the app to classify new messages. and a Python pickle file with input for some of the graphs in the app.

`app` contains the web application.

## Discussion of approach
### Imbalanced data
The data is imbalanced. There are few massages in the training data for some of the 36 categories to classify.

__Accuracy not appropriate:__<br>
Hence, we need to take care of how to measure the classification performance. When a category like 'water' appears just 1% of the time we are 99% right to not predict the category 'water' at all. It is the accuracy score giving 99%. Both recall and precision are helpful for imbalanced data. Recall answers how many of the values actually positive are identified correctly? Precision answers how many of the values predicted as positive are identified correctly?

__Maximizing the F2 score:__<br>
Looking for the model with the highest F1 score gives equal weight to maximize both recall and precision. For the disaster response model we want to consider precision but give more weight to recall:

We decide for a __recall-oriented model__ to accept to rather providing e.g. water too often as opposed to not help people who are in need. Thus we use the [F-beta scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) with a beta of 2, i.e. the F2 score. In this multilabel setting, we choose to maximize the F2 score average over all categories weighted by true instances for each label (`average='weighted'`).

### Machine learning pipeline runtime
The `train_classifier.py` is optimized to not take hours to run. A random sample of 5000 of the labeled messages is used.

You can change that in the Python file –look for `load_data(database_filepath, n_draws=5000)`. Also the number iteration and cross validation runs is set quite low searching for the best model. You can change that in the Python file – look for: ```best_model = search.train_model(X_train, y_train, search='random',
                scoring=scorer, n_iter=3, cv=2, iid=False)```

The F2 score obtained from a test run was .567.
