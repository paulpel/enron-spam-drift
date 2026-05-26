import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

N_SPLITS = 5
RANDOM_STATE = 42


def _bow_classifiers():
    """Classifiers for bag-of-words (count) features."""
    return {
        "Naive Bayes": MultinomialNB(),  # counts -> MultinomialNB is appropriate
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _bert_classifiers():
    """Classifiers for dense BERT embeddings.

    Uses ``GaussianNB`` rather than ``MultinomialNB`` because BERT embeddings are
    real-valued (and include negatives); ``MultinomialNB`` only accepts counts.
    """
    return {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _per_fold_accuracy(model, X, y, n_splits=N_SPLITS):
    """Accuracy of a fitted model on each stratified fold of (X, y).

    Yields a *distribution* of scores (one per fold) so the downstream paired
    t-tests have comparable samples instead of a single number.
    """
    y = np.asarray(y)
    y_pred = model.predict(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [accuracy_score(y[idx], y_pred[idx]) for _, idx in cv.split(X, y)]


def train_classifiers(data):
    """Train bag-of-words classifiers, scored with stratified k-fold CV.

    :param data: dataset with ``message`` and ``label`` columns.
    :type data: pandas.DataFrame
    :return: ``(trained_models, scores, vectorizer)`` — each ``scores[name]`` holds
        the mean accuracy and the per-fold scores used for statistical testing.
    :rtype: tuple(dict, dict, CountVectorizer)
    """
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    X = vectorizer.fit_transform(data["message"])
    y = data["label"]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    trained_models = {}
    scores = {}
    for name, clf in _bow_classifiers().items():
        fold_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        clf.fit(X, y)  # refit on all data for later evaluation on the drift set
        trained_models[name] = clf
        scores[name] = {
            "accuracy": float(fold_scores.mean()),
            "fold_scores": fold_scores.tolist(),
        }
        print(f"{name}: CV accuracy {fold_scores.mean():.4f} (+/- {fold_scores.std():.4f})")

    return trained_models, scores, vectorizer


def evaluate_models(trained_models, vectorizer, data):
    """Evaluate already-trained models on a new (e.g. drifted) dataset, per fold.

    :return: ``scores`` dict with mean accuracy and per-fold scores per classifier.
    :rtype: dict
    """
    X = vectorizer.transform(data["message"])
    y = data["label"]

    scores = {}
    for name, model in trained_models.items():
        fold_scores = _per_fold_accuracy(model, X, y)
        scores[name] = {
            "accuracy": float(np.mean(fold_scores)),
            "fold_scores": fold_scores,
        }
        print(f"{name}: drift accuracy {np.mean(fold_scores):.4f}")

    return scores


def extract_features_with_bert(
    data, model_name="bert-base-uncased", max_length=128, batch_size=32
):
    """
    Extract features from text data using a pre-trained BERT model.

    This function tokenizes and encodes the text data using a BERT tokenizer, processes it in batches,
    and extracts features using the BERT model.

    :param data: The dataset containing messages.
    :type data: pandas.DataFrame
    :param model_name: The name of the pre-trained BERT model to use.
    :type model_name: str
    :param max_length: The maximum length of the tokenized sequences.
    :type max_length: int
    :param batch_size: The batch size for processing the data.
    :type batch_size: int
    :return: A numpy array containing the extracted features.
    :rtype: np.ndarray
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    # Tokenize and encode the messages
    encoded_inputs = tokenizer(
        data["message"].tolist(),  # Directly use the list of strings
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create a DataLoader for the encoded inputs
    dataset = torch.utils.data.TensorDataset(
        encoded_inputs["input_ids"], encoded_inputs["attention_mask"]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Extract features
    features = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting BERT features"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the CLS token embedding as the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_embeddings)

    # Concatenate all features into a single array
    features = np.concatenate(features, axis=0)

    return features


def train_classifiers_with_bert_features(data, features):
    """Train classifiers on BERT-extracted features, scored with stratified k-fold CV.

    :param data: dataset containing the ``label`` column.
    :type data: pandas.DataFrame
    :param features: BERT-extracted features.
    :type features: np.ndarray
    :return: ``(trained_models, scores)`` with mean + per-fold accuracy per classifier.
    :rtype: tuple(dict, dict)
    """
    y = data["label"].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    trained_models = {}
    scores = {}
    for name, clf in _bert_classifiers().items():
        fold_scores = cross_val_score(clf, features, y, cv=cv, scoring="accuracy")
        clf.fit(features, y)
        trained_models[name] = clf
        scores[name] = {
            "accuracy": float(fold_scores.mean()),
            "fold_scores": fold_scores.tolist(),
        }
        print(f"{name}: BERT CV accuracy {fold_scores.mean():.4f} (+/- {fold_scores.std():.4f})")

    return trained_models, scores


def bert_evaluate_models(trained_models, features, data):
    """Evaluate trained classifiers on BERT features of a new dataset, per fold.

    :return: ``scores`` dict with mean accuracy and per-fold scores per classifier.
    :rtype: dict
    """
    y = data["label"].values

    scores = {}
    for name, model in trained_models.items():
        fold_scores = _per_fold_accuracy(model, features, y)
        scores[name] = {
            "accuracy": float(np.mean(fold_scores)),
            "fold_scores": fold_scores,
        }
        print(f"{name}: BERT drift accuracy {np.mean(fold_scores):.4f}")

    return scores
