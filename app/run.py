"""
Flask web application for the Disaster Response Pipeline project.

Usage:
    Desde la raiz del proyecto:
    python app\run.py
    Luego abrir: http://localhost:3001
"""

import json
import os
import pickle
import re
import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine

app = Flask(__name__)

# Ruta absoluta al DB y al modelo, sin importar desde donde se ejecute
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, '..', 'data', 'DisasterResponse.db')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'classifier.pkl')


def tokenize(text):
    """
    Tokenize and clean a text string.
    Must match the tokenizer used during training.
    """
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'not', 'no', 'so', 'as', 'if', 'than', 'then', 'when', 'where'
    }
    text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text).lower())
    tokens = text.split()
    return [t for t in tokens if t not in stop_words and len(t) > 2]



# Load data and model

engine = create_engine(f'sqlite:///{DB_PATH}')
df = pd.read_sql_table('DisasterResponse', engine)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)



# Routes


@app.route('/')
@app.route('/index')
def index():
    """Render the main dashboard with three Chart.js visualizations."""

    genre_counts = df.groupby('genre').count()['message']
    genre_names  = list(genre_counts.index)
    genre_values = [int(v) for v in genre_counts.values]

    category_cols  = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_totals = category_cols.sum().sort_values(ascending=False).head(10)
    top_categories  = list(category_totals.index)
    top_counts      = [int(v) for v in category_totals.values]

    chart_data = {
        'genre_names':    genre_names,
        'genre_values':   genre_values,
        'top_categories': top_categories,
        'top_counts':     top_counts,
    }

    return render_template('master.html', chart_data=json.dumps(chart_data))


@app.route('/go')
def go():
    """Classify a user message and show results."""
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    category_cols = df.drop(columns=['id', 'message', 'original', 'genre'])
    classification_results = dict(zip(category_cols.columns, classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
