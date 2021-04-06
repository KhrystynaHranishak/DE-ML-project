import preprocessing as prp
from config import *
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn import feature_selection, metrics
import pandas as pd
import numpy as np
import joblib
import sys


def upload_train_data(path):
    return pd.read_csv(path)


def is_data_format_correct(data):

    if 'comment_text' in data.columns and 'toxic' in data.columns:
        return True
    else:
        return False


def select_features(data, tf_idf_vecrorizer):

    y = data['toxic']
    corpus = data["text_clean"]
    tf_idf_vecrorizer.fit(corpus)
    features_potential = vectorizer.transform(corpus)
    feature_names = tf_idf_vecrorizer.get_feature_names()
    p_value_limit = 0.95
    df_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(features_potential, y == cat)
        df_features = df_features.append(pd.DataFrame(
            {"feature": feature_names, "score": 1 - p, "y": cat}))
        df_features = df_features.sort_values(["y", "score"],
                                              ascending=[True, False])
        df_features = df_features[df_features["score"] > p_value_limit]
    feature_names = df_features["feature"].unique().tolist()

    return feature_names


def print_metrics_details(ground_truth, predictions_labels, predictions_prob, plot=False):

    classes = np.unique(ground_truth)
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(ground_truth, predictions_labels)
    auc = metrics.roc_auc_score(ground_truth, predictions_prob)
    print("Accuracy:", round(accuracy, 2))
    print("AUC:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(ground_truth, predictions_labels))

    ## Plot confusion matrix
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set_palette("pastel")

        cm = metrics.confusion_matrix(ground_truth, predictions_labels)
        fig, ax = plt.subplots()
        _ = sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                    cbar=False)
        ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
        plt.yticks(rotation=0)

        plt.show()


if __name__ == '__main__':
    print('Uploading training data...')
    train_data = upload_train_data(TRAIN)
    if is_data_format_correct(train_data):
        print('Data is uploaded successfully')
    else:
        print('Check the data format. It should contain columns `comment_text` and `toxic`')
        sys.exit()
    # text preprocessing
    train_data["text_clean"] = train_data['comment_text'].apply(prp.utils_preprocess_text,
                                                                lst_stopwords=prp.stopwords_en)
    # train-test split
    dtf_train, dtf_val = model_selection.train_test_split(train_data, test_size=0.3, random_state=42,
                                                          stratify=train_data["toxic"])
    y_train = dtf_train["toxic"].values
    y_val = dtf_val["toxic"].values

    print('Train size: ', dtf_train.shape[0])
    print('Test size: ', dtf_val.shape[0])
    print('\n')
    print('Train true positive rate:', dtf_train[dtf_train["toxic"] == 1].shape[0] / dtf_train.shape[0])
    print('Test true positive rate:', dtf_val[dtf_val["toxic"] == 1].shape[0] / dtf_val.shape[0])

    ## Tf-Idf (advanced variant of BoW)
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    corpus = dtf_train["text_clean"]
    vectorizer.fit(corpus)
    features_important = select_features(dtf_train, vectorizer)
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=features_important)
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    classifier = naive_bayes.MultinomialNB()
    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])
    ## train classifier
    model["classifier"].fit(X_train, y_train)

    ## validate a model
    X_val = dtf_val["text_clean"].values
    predicted_val = model.predict(X_val)
    predicted_prob_val = model.predict_proba(X_val)[:, 1]

    predicted_train = model["classifier"].predict(X_train)
    predicted_prob_train = model["classifier"].predict_proba(X_train)[:, 1]
    print('Train metrics')
    print_metrics_details(y_train, predicted_train, predicted_prob_train)
    print('Validation metrics')
    print_metrics_details(y_val, predicted_val, predicted_prob_val)

    ## save a model
    joblib.dump(model, 'Data/'+MODEL_NAME)