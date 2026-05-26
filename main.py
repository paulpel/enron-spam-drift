import os
import pandas as pd
import numpy as np

from train_evaluate import (
    train_classifiers,
    evaluate_models,
    extract_features_with_bert,
    train_classifiers_with_bert_features,
    bert_evaluate_models,
)
from preprocess_data import prepare_data, bert_prepare_data
from statistical_tests import compare_classifiers


def main(original_data_path, drifted_data_path, output_folder):
    """
    Main function to train and evaluate classifiers on both regular and BERT-encoded data.

    This function performs the following steps:
    1. Loads or prepares the regular and drifted datasets.
    2. Loads or prepares the BERT-encoded datasets.
    3. Swaps the order of dataframes to better match test data length.
    4. Trains classifiers on the regular CSV data.
    5. Evaluates classifiers on the drifted CSV data.
    6. Extracts features using BERT.
    7. Trains classifiers on the BERT features.
    8. Evaluates classifiers on the BERT features.

    :param original_data_path: Path to the original (non-drifted) data CSV file.
    :type original_data_path: str
    :param drifted_data_path: Path to the drifted data CSV file.
    :type drifted_data_path: str
    :param output_folder: Path to the output folder where results will be saved.
    :type output_folder: str
    """
    if os.path.exists("enron_spam_data.pkl") and os.path.exists("processed_data.pkl"):
        regular_df = pd.read_pickle("enron_spam_data.pkl")
        drifted_df = pd.read_pickle("processed_data.pkl")
    else:
        regular_df, drifted_df = prepare_data(original_data_path, drifted_data_path)

    if os.path.exists("bert_enron_spam_data.pkl") and os.path.exists(
        "bert_processed_data.pkl"
    ):
        bert_regular_df = pd.read_pickle("bert_enron_spam_data.pkl")
        bert_drifted_df = pd.read_pickle("bert_processed_data.pkl")
    else:
        bert_regular_df, bert_drifted_df = bert_prepare_data(
            original_data_path, drifted_data_path
        )

    # Swap the order of the dataframes to better match test data length (might be temporary)
    regular_df, drifted_df = drifted_df, regular_df
    bert_regular_df, bert_drifted_df = bert_drifted_df, bert_regular_df

    # Step 1: Train classifiers on regular CSV data
    print("Training initial classifiers on regular data...")
    trained_models, scores, vectorizer = train_classifiers(regular_df)

    # Step 2: Evaluate classifiers on drifted CSV data
    print("Evaluating classifiers on drifted data...")
    drifted_scores = evaluate_models(trained_models, vectorizer, drifted_df)

    # Step 3: Extract features with BERT
    print("Extracting features with BERT...")
    regular_features, drifted_features = None, None
    if os.path.exists("bert_encoded_features_original.csv") and os.path.exists(
        "bert_encoded_features_drifted.csv"
    ):
        regular_features = np.loadtxt(
            "bert_encoded_features_original.csv", delimiter=","
        )
        drifted_features = np.loadtxt(
            "bert_encoded_features_drifted.csv", delimiter=","
        )
    else:
        regular_features = extract_features_with_bert(bert_regular_df)
        drifted_features = extract_features_with_bert(bert_drifted_df)
        np.savetxt(
            "bert_encoded_features_original.csv", regular_features, delimiter=","
        )
        np.savetxt("bert_encoded_features_drifted.csv", drifted_features, delimiter=",")

    # Step 4: Train classifiers on BERT features
    # (GaussianNB handles the real-valued BERT embeddings directly, so no
    # offset hack is needed to keep features non-negative.)
    print("Training classifiers on BERT features on regular data...")
    bert_trained_models, bert_scores = train_classifiers_with_bert_features(
        regular_df, regular_features
    )

    # Step 5: Evaluate classifiers on BERT features
    print("Evaluating classifiers on BERT features on drifted data...")
    bert_drifted_scores = bert_evaluate_models(
        bert_trained_models, drifted_features, drifted_df
    )

    # Step 6: Performe statistical tests on scores for different data
    # Compare Normal vs Drifted Data Scores
    comparison_normal_vs_drifted = compare_classifiers(scores, drifted_scores, label1='Normal', label2='Drifted')
    comparison_normal_vs_drifted.to_csv(os.path.join(output_folder, "comparison_normal_vs_drifted.csv"), index=False)
    print("Comparison (Normal vs Drifted):\n", comparison_normal_vs_drifted)

    # Compare Normal Data vs BERT Features
    comparison_normal_vs_bert = compare_classifiers(scores, bert_scores, label1='Normal', label2='BERT')
    comparison_normal_vs_bert.to_csv(os.path.join(output_folder, "comparison_normal_vs_bert.csv"), index=False)
    print("Comparison (Normal vs BERT):\n", comparison_normal_vs_bert)

    # Compare Drifted Data vs BERT Features
    comparison_drifted_vs_bert = compare_classifiers(drifted_scores, bert_drifted_scores, label1='Drifted', label2='BERT')
    comparison_drifted_vs_bert.to_csv(os.path.join(output_folder, "comparison_drifted_vs_bert.csv"), index=False)
    print("Comparison (Drifted vs BERT):\n", comparison_drifted_vs_bert)

    # Compare Normal Data vs Drifted Data using BERT
    comparison_bert_normal_vs_drifted = compare_classifiers(bert_scores, bert_drifted_scores, label1='BERT Normal', label2='BERT Drifted')
    comparison_bert_normal_vs_drifted.to_csv(os.path.join(output_folder, "comparison_bert_normal_vs_drifted.csv"), index=False)
    print("Comparison (BERT Normal vs Drifted):\n", comparison_bert_normal_vs_drifted)


if __name__ == "__main__":
    main("enron_spam_data.csv", "processed_data.csv", "")
