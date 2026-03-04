# Disaster Response Pipeline

A machine learning pipeline that classifies disaster-related messages into 36 categories to help emergency response organizations prioritize and route aid requests effectively.

---

## Project Structure

```
disaster_response_pipeline/
│
├── app/
│   ├── templates/
│   │   ├── master.html        # Main page of the web app
│   │   └── go.html            # Classification results page
│   └── run.py                 # Flask application
│
├── data/
│   ├── disaster_messages.csv      # Raw messages dataset
│   ├── disaster_categories.csv    # Raw categories dataset
│   ├── process_data.py            # ETL pipeline script
│   └── DisasterResponse.db        # Cleaned data stored in SQLite
│
├── models/
│   ├── train_classifier.py        # ML pipeline script
│   └── classifier.pkl             # Trained model (generated after training)
│
└── README.md
```

---

## Installation

Install required dependencies:

```bash
pip install pandas sqlalchemy scikit-learn nltk flask plotly
```

---

## How to Run

### 1. ETL Pipeline — Clean and store the data


```bash
 python data\process_data.py data\messages.csv data\categories.csv data\DisasterResponse.db
```

This will:
- Load and merge both CSV datasets
- Clean the data (split categories, binarize values, drop duplicates)
- Store the result in `DisasterResponse.db`

---

### 2. ML Pipeline — Train and save the classifier


```bash
python models\train_classifier.py data\DisasterResponse.db models\classifier.pkl
```

This will:
- Load the cleaned data from the database
- Train a `RandomForestClassifier` inside a text-processing pipeline using TF-IDF
- Optimize hyperparameters with `GridSearchCV`
- Print precision, recall and F1-score for all 36 categories on the test set
- Save the trained model as `classifier.pkl`

---

### 3. Web App — Launch the Flask application

From the `app/` directory:

```bash
python run.py
```

Then open your browser at `http://localhost:3001`.

Features:
- Enter any disaster message and get instant classification across 36 categories
- Three Plotly visualizations describing the training data:
  1. Distribution of message genres (Bar chart)
  2. Top 10 most frequent categories (Bar chart)
  3. Genre share of all messages (Pie chart)

---

## ETL Pipeline Details (`process_data.py`)

| Step | Description |
|------|-------------|
| Load | Reads `disaster_messages.csv` and `disaster_categories.csv` |
| Merge | Joins both datasets on the `id` column |
| Split | Expands the `categories` column into 36 individual binary columns |
| Binarize | Converts category values to 0/1; clips values > 1 to 1 |
| Deduplicate | Removes duplicate rows |
| Store | Saves to SQLite via `pandas.to_sql()` |

---

## ML Pipeline Details (`train_classifier.py`)

| Component | Detail |
|-----------|--------|
| Tokenizer | Custom `tokenize()`: lowercase → remove punctuation → tokenize → remove stopwords → lemmatize |
| Vectorizer | `CountVectorizer` using the custom tokenizer |
| TF-IDF | `TfidfTransformer` |
| Classifier | `MultiOutputClassifier(RandomForestClassifier)` |
| Tuning | `GridSearchCV` over `n_estimators` and `min_samples_split` |
| Evaluation | `classification_report` (precision, recall, F1) per category |

---

## License

This project was developed as part of the Udacity Data Science Nanodegree Program.
