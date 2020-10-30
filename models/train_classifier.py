# ML pipeline
# example use in comandline: python train_classifier.py ../data/DisasterResponse.db disaster_model.pkl

import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, make_scorer, fbeta_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC
import pickle

# suppress warnings for the user # https://docs.python.org/3/library/warnings.html
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def load_data(database_filepath, n_draws=None):
    '''
    Load data from a DB, specifically from a disaster.db the table 'disaster_table'.

    Input:
    n_draws (optional): sample draws from message database; can be used to run
                        the pipeline with a reduced workload for estimation
    Output:
    X: a Pandas series as model independent variable
    Y: a Pandas dataframe as model dependent variables
    '''
    engine = create_engine('sqlite:///' + os.path.join(os.getcwd(), database_filepath))
        # alternative to database_filepath input with ".." is to use os.pardir

    df = pd.read_sql_table('disaster_table', con=engine)

    # using the message column to predict 36 categories
    category_names = ['related', 'request', 'offer',
               'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
               'security', 'military', 'child_alone', 'water', 'food', 'shelter',
               'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
               'infrastructure_related', 'transport', 'buildings', 'electricity',
               'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
               'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
               'other_weather', 'direct_report']
    X = df.message
    y = df[category_names]

    if n_draws:
        X = X.sample(n=n_draws) # default is no replacement
        y = y.loc[X.index]      # same rows from X and y
    return X, y, category_names

def tokenize(text, stop_words_remove=True):
    '''
    Tokenize text messsages and cleaning for Machine Learning use.

    Input: str
    Output: list
    '''
    text = re.sub(r'[^A-Za-z0-9]', ' ', text.lower())
    words = word_tokenize(text)
    if stop_words_remove:
        result = [WordNetLemmatizer().lemmatize(w) for w in words
                  if w not in stopwords.words('english')]
    if stop_words_remove==False:
        result = [WordNetLemmatizer().lemmatize(w) for w in words]
    return result

class HyperclassifierTuning:
    # based on code from: https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb
    # the code was rewritten
    #
    # my documentation enhancements:
    #   added code documentation including docstrings
    #
    # my functionality enhancements:
    #   added option to use RandomizedSearchCV
    #   the best overall model is provided by train_model()
    #   output dataframe is simplified as standard option
    #
    # Code is also released as a package but not used here for simpler Udacity project review:
    # https://github.com/dabln/HyperclassifierSearch
    """Train multiple classifiers/pipelines with GridSearchCV or RandomizedSearchCV.

    HyperclassifierTuning implements a "train_model" and "evaluate_model" method.

    "train_model" returns the optimal model according to the scoring metric.

    "evaluate_model" gives the results for all classifiers/pipelines.

    Example usage:
    ----
    Example code:
    from sklearn import datasets
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    models = {
        'LogisticRegression': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression())
        ]),
        'linearSVC': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LinearSVC())
        ])
    }
    params = {
        'LogisticRegression': { 'clf__C': [0.1, 1] },
        'linearSVC': { 'clf__C': [1, 10, 100] }
    }

    search = HyperclassifierTuning(models, params)
    search.train_model(X, y)
    search.evaluate_model()
    """
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.grid_results = {}

    def train_model(self, X_train, y_train, search='grid', **search_kwargs):
        """
        Optimizing over multiple classifiers or pipelines.

        Input:
        X : array or dataframe with features; this should be a training dataset
        y : array or dataframe with label(s); this should be a training dataset

        Output:
        returns the optimal model according to the scoring metric

        Parameters:
        search : str, default='grid'
            define the search
            ``grid`` performs GridSearchCV
            ``random`` performs RandomizedSearchCV

        **search_kwargs : kwargs
            additional parameters passed to the search
        """
        grid_results = {}
        best_score = 0

        for key in self.models.keys():
            print('GridSearchCV for {}'.format(key), '...')
            assert search in ('grid', 'random'), 'search parameter out of range'
            if search=='grid':
                grid = GridSearchCV(self.models[key], self.params[key], **search_kwargs)
            if search=='random':
                grid = RandomizedSearchCV(self.models[key], self.params[key], **search_kwargs)
            grid.fit(X_train, y_train)
            self.grid_results[key] = grid

            if grid.best_score_ > best_score: # to return best model
                best_score = grid.best_score_
                best_model = grid

        print('Search is done.')
        return best_model # allows to predict with the best model overall

    def evaluate_model(self, sort_by='mean_test_score', show_timing_info=False):
        """
        Provides sorted model results for multiple classifier or pipeline runs of
        GridSearchCV or RandomizedSearchCV.

        Input: Fitted search object (accesses cv_results_).
        Output: Dataframe with a line for each training run including estimator name, parameters, etc.
        Parameters:
        sort_by: the metric to rank the model results
        """
        results = []
        for key, result in self.grid_results.items():
            print('results round for:', key)
            # get rid of column specific to estimator to use df for multiple estimators
            # regex 'not in': https://stackoverflow.com/questions/1971738/regex-for-all-strings-not-containing-a-string#1971762
            result = pd.DataFrame(result.cv_results_).filter(regex='^(?!.*param_).*')
            if show_timing_info==False: # skip timing info
                result = result.filter(regex='^(?!.*time).*')
            # add column with the name of the estimator
            result = pd.concat((pd.DataFrame({'Estimator': [key] * result.shape[0] }), result), axis=1)
            results.append(result)

        # handle combined classifier results:
        # sort by target metric and remove subset rank scores
        df_results = pd.concat(results).sort_values([sort_by], ascending=False).\
                        reset_index().drop(columns = ['index', 'rank_test_score'])
        return df_results


def build_model():
    """
    Defining the input for the HyperclassifierTuning class.

    Input:  None
    Output: model definition, parameter space to evaluate, scorer
    """
    models = {
        'AdaBoost': Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ]),
        'RandomForest': Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]),
        # skipped parts to allow for shorter runtime of the pipeline
        #'LinearSVC': Pipeline([
        #    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        #    ('clf', MultiOutputClassifier(LinearSVC(max_iter=100, class_weight='balanced')))
        #])
        #'GradientBoosting': Pipeline([
        #    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        #    ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
        #]),
        #'LogisticRegression': Pipeline([
        #    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        #    ('clf', MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=10000, random_state=42)))
        #])
    }
    params = {
        'AdaBoost': { 'clf__estimator__n_estimators': np.arange(16,32+1) },
        'RandomForest': { 'clf__estimator__n_estimators': np.arange(7,200+1) }
        # skipped parts to allow for shorter runtime of the pipeline
        #'LinearSVC': { 'clf__estimator__C': np.linspace(0.1, 100, num=1000) }
        #'GradientBoosting' : { 'clf__estimator__learning_rate': [0.8, 1.0] },
        #'LogisticRegression': { 'clf__estimator__C': [0.1, 1] }
    }
    #scorer = make_scorer(balanced_accuracy_score)
    scorer = make_scorer(fbeta_score, beta=2, average='weighted')
    return models, params, scorer

def evaluate_model(model, X_test, y_test, category_names):
    """
    Provide model results in a table separated for column classes.

    Input:
    - model: fitted model to use
    - X_test: features from the test set
    - y_test: labels from test dataset
    - category_names: 36 category labels

    Output:
    - [0]: dataframe with evaluation eparated for column classes
    - [1]: dataframe with mean column values of [0]
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names
    report = {}
    for col in y_pred_df.columns:
        output = classification_report(y_test[col], y_pred_df[col], output_dict=True)
        report[col] = {} # inspired by https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python
        for i in output:
            if i == 'accuracy':
                break
            report[col]['f1_' + i] = output[i]['f1-score']
            report[col]['precision_' + i] = output[i]['precision']
            report[col]['recall_' + i] = output[i]['recall']
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.columns.sort_values()]
    report_df_mean = report_df.mean()
    print(report_df)
    return report_df, report_df_mean

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

# Data for the app: scatter plot for word count per message
def save_scatter_data(X):
    # message lengths in terms of words including stop words
    message_words_stopwords = X.apply(lambda x: tokenize(x, stop_words_remove=False)).apply(len).value_counts()
    # message lengths in words without stop words
    message_words_no_stopwords = X.apply(tokenize).apply(len).value_counts()
    messages = pd.DataFrame(message_words_stopwords).reset_index()
    messages['type'] = 'including stopwords'
    messages2 = pd.DataFrame(message_words_no_stopwords).reset_index()
    messages2['type'] = 'without stopwords'
    message_scatter_data = pd.concat([messages, messages2])
    # using a fixed file path is favored such that running the pipeline is
    # not complicated in terms of expecting more parameters
    with open('../data/scatter_data.pkl', 'wb') as file:
        pickle.dump(message_scatter_data, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # not all messages are used for classification to allow for reasonable
        # runtimes of the pipeline on an average local machine
        X, y, category_names = load_data(database_filepath, n_draws=5000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        models, params, scorer = build_model()

        print('Training model...')
        search = HyperclassifierTuning(models, params)

        # cv and n_iter values are set quite low to allow for reasonable
        # runtimes on an average local machine
        # for more reliable results: set cv = 10, n_iter=10
        best_model = search.train_model(X_train, y_train, search='random',
                        scoring=scorer, n_iter=2, cv=2, iid=False)

        print('Evaluating model...')
        # not using HyperclassifierTuning evaluation given above as it does not split by categories
        evaluate_model(best_model, X_test, y_test, category_names)[0]

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)
        print('Trained model saved!')

        print('Saving plot specific data...\n    TO FILE: scatter_data.pkl')
        save_scatter_data(X)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db '\
              'disaster_model.pkl')

if __name__ == '__main__':
    main()
