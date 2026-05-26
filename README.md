# Enron Spam Detection

This project aims to train and evaluate classifiers for detecting spam emails using both regular and BERT-extracted features. The project consists of multiple scripts that handle data preprocessing, feature extraction, training classifiers, and evaluating their performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [Main Script](#main-script)
  - [Classifier Training and Evaluation](#classifier-training-and-evaluation)
  - [Data Preprocessing](#data-preprocessing)
  - [Statistical Tests](#statistical-tests)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/paulpel/enron-spam-drift.git
    cd enron-spam-drift
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download NLTK data:
    ```python
    python -m nltk.downloader punkt stopwords wordnet
    ```

## Usage

To run the main script and train the classifiers (the last argument is the
output folder for the comparison CSVs — use `.` for the current directory):

```bash
python main.py enron_spam_data.csv processed_data.csv .
```

Each classifier is scored with 5-fold stratified cross-validation, and the
conditions (normal vs drifted, bag-of-words vs BERT) are compared with paired
t-tests over the per-fold scores.

## Scripts Overview

### Main Script

**File:** `main.py`

This script orchestrates the entire process of loading data, training classifiers, and evaluating their performance. It handles both regular CSV data and BERT-extracted features.

**Functions:**
- `main(original_data_path, drifted_data_path, output_folder)`: Main function that coordinates data loading, preprocessing, training, and evaluation.

### Classifier Training and Evaluation

**File:** `train_evaluate.py`

This script contains functions for training classifiers on the provided datasets and evaluating their performance.

**Functions:**
- `train_classifiers(data)`: Trains classifiers on the given dataset.
- `evaluate_models(trained_models, vectorizer, data)`: Evaluates the trained classifiers on a new dataset.
- `extract_features_with_bert(data, model_name, max_length, batch_size)`: Extracts features from text data using a pre-trained BERT model.
- `train_classifiers_with_bert_features(data, features)`: Trains classifiers on BERT-extracted features.
- `bert_evaluate_models(trained_models, features, data)`: Evaluates the trained classifiers on BERT-extracted features.

### Data Preprocessing

**File:** `preprocess_data.py`

This script contains functions for preprocessing and tokenizing the data for both regular and BERT-based feature extraction.

**Functions:**
- `tokenize_dataset_enron(data)`: Tokenizes and preprocesses the Enron dataset.
- `tokenize_dataset_processed(data)`: Tokenizes and preprocesses the processed dataset.
- `bert_tokenize_dataset_enron(data)`: Tokenizes the Enron dataset for BERT.
- `bert_tokenize_dataset_processed(data)`: Tokenizes the processed dataset for BERT.
- `prepare_data(data_path, drift_data_path)`: Prepares and tokenizes the Enron and processed datasets, and saves them as pickle files.
- `bert_prepare_data(data_path, drift_data_path)`: Prepares and tokenizes the Enron and processed datasets for BERT, and saves them as pickle files.

### Statistical Tests

**file** `statistical_tests.py`

This script contains functions for comparing the performance of different classifiers using statistical tests.

**Functions:**
- `compare_classifiers(scores1, scores2, metric='accuracy', label1='Set 1', label2='Set 2')`: Compares classifiers using paired t-tests on the specified metric.

