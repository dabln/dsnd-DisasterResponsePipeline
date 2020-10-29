import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib # https://stackoverflow.com/questions/61893719/importerror-cannot-import-name-joblib-from-sklearn-externals
import joblib
from sqlalchemy import create_engine

from wrangle_graphs import return_graphs

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('disaster_table', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    number_lines = df.shape[0]
    # # extract data for visuals
    # categories = df.iloc[:,4:40].sum().sort_values(ascending=False)
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    #
    # number_lines = df.shape[0]
    #
    # import plotly.express as px
    # df1 = px.data.iris()
    # scatter_x = df1.sepal_width
    # scatter_y = df1.sepal_length
    #fig = px.scatter(df1, x="sepal_width", y="sepal_length", color="species",
    #                 size='petal_length', hover_data=['petal_width'])

    #fig.show()

    # create visuals

    #import plotly.graph_objs as go

    # Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
    # `data/file_name.csv`

    # def return_graphs():
    #     """Creates four plotly visualizations
    #
    #     Args:
    #         None
    #
    #     Returns:
    #         list (dict): list containing the four plotly visualizations
    #
    #     """
    #
    #     graph_one = []
    #
    #     graph_one.append(
    #       go.Bar(
    #       x = categories.index,
    #       y = categories,
    #       )
    #     )
    #
    #     layout_one = dict(title = 'Distribution of Message Categories',
    #                 xaxis = dict(title = 'Category',),
    #                 yaxis = dict(title = 'Count'),
    #                 )
    #
    #     graph_two = []
    #
    #     graph_two.append(
    #       go.Bar(
    #       x = genre_names,
    #       y = genre_counts,
    #       )
    #     )
    #
    #     layout_two = dict(title = 'Distribution of Message Genres',
    #                 xaxis = dict(title = 'Genre',),
    #                 yaxis = dict(title = 'Count'),
    #                 )
    #
    #
    #     graph_three = []
    #     graph_three.append(
    #       go.Scatter(
    #       x = scatter_x,
    #       y = scatter_y,
    #       mode = 'lines'
    #       )
    #     )
    #
    #     layout_three = dict(title = 'Chart One',
    #                 xaxis = dict(title = 'x-axis label'),
    #                 yaxis = dict(title = 'y-axis label'),
    #                 )
    #
    #     # append all charts to the figures list
    #     graphs = []
    #     graphs.append(dict(data=graph_one, layout=layout_one))
    #     graphs.append(dict(data=graph_two, layout=layout_two))
    #     graphs.append(dict(data=graph_three, layout=layout_three))
    #
    #     return graphs
    graphs = return_graphs(df)

    # graphs = [
    #
    #     {
    #         'data': [
    #             Bar(
    #                 x=categories.index,
    #                 y=categories
    #             )
    #         ],
    #
    #         'layout': {
    #             'title': 'Distribution of Message Categories',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Category"
    #             }
    #         }
    #     },
    #
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],
    #
    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    #
    #
    # ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,
        number_lines=number_lines)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
