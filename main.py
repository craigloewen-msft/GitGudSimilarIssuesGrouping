from pymongo import MongoClient
from pinecone import Pinecone
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from tkinter import Tk, Canvas, Frame, Scrollbar
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.io as pio

repoShortURL = 'microsoft/powertoys'
startingQueryDate = datetime(2024, 1, 1)
numberOfClusters = 7

# Create Dash application
app = dash.Dash(__name__)

# Read configurations from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Step 1: Connect to MongoDB and retrieve the repository ID and issue list
def get_repository_id_and_issue_list(repo_name):
    client = MongoClient(config['devMongoDBConnectionString'])
    db = client['GithubIssueManagement']
    
    print("Getting repo id and issue list")
    
    # Retrieve repository ID
    repo_collection = db['repoInfo']
    repo_result = repo_collection.find_one({'shortURL': repo_name})
    repo_id = repo_result['_id']

    # Retrieve issues created more than 3 years ago for the given repository
    issue_collection = db['issueInfo']
    issues = issue_collection.find({
        'repoRef': repo_id,
        'created_at': {'$gt': startingQueryDate}
    })

    # Change issue_list to a dictionary
    issue_dict = {str(issue['_id']): issue for issue in issues}

    return repo_id, issue_dict

# Step 2: Connect to Pinecone and retrieve all embedding values for each issue
def get_embedding_values(issue_list, repository_id):
    pinecone = Pinecone(api_key=config['pineconeAPIKey'])
    index = pinecone.Index(host=config['pineconeIndexURL'])

    batchSize = 1000
    batchArray = []

    for issue_id in issue_list:
        batchArray.append(issue_id)

        if len(batchArray) == batchSize:
            print("Fetching embeddings...")
            batch_embeddings = index.fetch(ids=batchArray, namespace=str(repository_id))

            for embedding_issue_id, embedding_value in batch_embeddings['vectors'].items():
                issue = issue_list[embedding_issue_id]
                issue['embedding'] = embedding_value.values

            batchArray = []

    if len(batchArray) > 0:
        print("Fetching embeddings...")
        batch_embeddings = index.fetch(ids=batchArray, namespace=str(repository_id))

        for embedding_issue_id, embedding_value in batch_embeddings['vectors'].items():
            issue = issue_list[embedding_issue_id]
            issue['embedding'] = embedding_value.values

    print("Done fetching embeddings!")

# Step 3: Perform hierarchical clustering on embeddings
def hierarchical_clustering(issue_list):
    # Extract embeddings from each issue in the dictionary
    embeddings = [issue['embedding'] for issue in issue_list.values()]

    clustering = AgglomerativeClustering(n_clusters=numberOfClusters)
    cluster_labels = clustering.fit_predict(embeddings)
    return embeddings, cluster_labels

# Step 4: Visualize the clustering in an interactive GUI window

def visualize_clustering(embeddingsList, cluster_labels, issue_list):
    tsne = TSNE(n_components=2)
    embeddings = np.array(embeddingsList)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Add issue title and html_url to DataFrame
    df = pd.DataFrame(embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df['Cluster'] = cluster_labels
    df['Title'] = [issue['title'] for issue in issue_list.values()]
    df['Number'] = [issue['number'] for issue in issue_list.values()]

    fig = px.scatter(df, x='Dimension 1', y='Dimension 2', color='Cluster', hover_data=['Title'], custom_data=['Number', 'Title'])

    fig.update_traces(
        hovertemplate="<b>%{customdata[1]}</b><br>Number: %{customdata[0]}",
        marker=dict(size=10)
    )

    fig.layout.hovermode = 'closest'
    fig.layout.clickmode = 'event+select'

    # Get unique clusters
    unique_clusters = df['Cluster'].unique()

    # Define layout
    app.layout = html.Div([
        dcc.Graph(figure=fig, id='main-graph'),
        html.Div([html.Button(f'Filter {cluster}', id=f'filter-{cluster}', n_clicks=0) for cluster in unique_clusters], id='filter-buttons'),
        html.Button('Clear', id='clear-button', n_clicks=0),
        html.Div(id='cluster-info'),
    ])

    # Define callback for filter buttons
    @app.callback(
        Output('main-graph', 'figure'),
        Output('cluster-info', 'children'),
        [Input(f'filter-{cluster}', 'n_clicks') for cluster in unique_clusters] +
        [Input('clear-button', 'n_clicks')],
        [State('main-graph', 'figure')]
    )
    def update_graph(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if 'clear' in button_id:
            fig = px.scatter(df, x='Dimension 1', y='Dimension 2', color='Cluster', hover_data=['Title'], custom_data=['Number', 'Title'])
        else:
            cluster = int(button_id.split('-')[1])
            df_filtered = df[df['Cluster'] == cluster]
            fig = px.scatter(df_filtered, x='Dimension 1', y='Dimension 2', color='Cluster', hover_data=['Title'], custom_data=['Number', 'Title'])

        # Calculate number of issues in each cluster
        cluster_counts = df['Cluster'].value_counts().to_dict()
        cluster_info = [html.P(f'Cluster {cluster}: {count} issues') for cluster, count in cluster_counts.items()]

        pio.write_html(fig, 'plot.html')

        return fig, cluster_info

    app.run_server(debug=False)

# Main function
def main():
    repository_name = repoShortURL
    repository_id, issue_list = get_repository_id_and_issue_list(repository_name)
    get_embedding_values(issue_list, repository_id)
    embeddings, cluster_labels = hierarchical_clustering(issue_list)
    visualize_clustering(embeddings, cluster_labels, issue_list)

if __name__ == "__main__":
    main()
