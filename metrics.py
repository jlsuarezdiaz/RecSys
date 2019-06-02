import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def coverage(predicted, catalog):
    """
    Computes the coverage for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    Returns
    ----------
    coverage:
        The coverage of the recommendations as a percent
        rounded to 2 decimal places
    """
    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_predictions = len(set(predicted_flattened))
    # coverage = round(unique_predictions / (len(catalog) * 1.0) * 100, 2)
    coverage = unique_predictions / len(catalog)
    return coverage


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def personalization(predicted):
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.
    Parameters:
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        The personalization score for all recommendations.
    """

    def make_rec_matrix(predicted, unique_recs):
        rec_matrix = pd.DataFrame(index=range(len(predicted)), columns=unique_recs)
        rec_matrix.fillna(0, inplace=True)
        for i in rec_matrix.index:
            rec_matrix.loc[i, predicted[i]] = 1
        return rec_matrix

    # get all unique items recommended
    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_recs = list(set(predicted_flattened))

    # create matrix for recommendations
    rec_matrix = make_rec_matrix(predicted, unique_recs)
    rec_matrix_sparse = sp.csr_matrix(rec_matrix.values)

    # calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

    # get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    # calculate average similarity
    personalization = np.mean(similarity[upper_right])
    return 1 - personalization


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def _single_list_similarity(predicted, feature_df):
    """
    Computes the intra-list similarity for a single list of recommendations.
    Parameters
    ----------
    predicted : a list
        Ordered predictions
        Example: ['X', 'Y', 'Z']
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
    ils_single_user: float
        The intra-list similarity for a single list of recommendations.
    """
    # get features for all recommended items
    # recs_content = feature_df.loc[predicted]
    # recs_content = recs_content.dropna()
    # print(recs_content)
    # recs_content = sp.csr_matrix(recs_content.values)
    recs_content = feature_df[predicted, :]

    # calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    # get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    # calculate average similarity score of all recommended items in list
    ils_single_user = np.mean(similarity[upper_right])
    return ils_single_user


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def intra_list_similarity(predicted, feature_df):
    """
    Computes the average intra-list similarity of all recommendations.
    This metric can be used to measure diversity of the list of recommended items.
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        Example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
        The average intra-list similarity for recommendations.
    """
    # feature_df = feature_df.fillna(0)
    Users = range(len(predicted))
    ils = [_single_list_similarity(predicted[u], feature_df) for u in Users]
    return np.mean(ils)


# Cálculo de la novedad de una lista de recomendaciones.
def novelty(predicted):
    # get all unique items recommended
    predicted_flattened = np.array([p for sublist in predicted for p in sublist])
    unique_recs = list(set(predicted_flattened))

    probs = np.array([sum(predicted_flattened == u) / len(predicted) for u in unique_recs])
    user_probs = [probs[np.where(np.isin(unique_recs, predicted[u]))] for u in range(len(predicted))]
    novelty = - np.mean([np.sum(p * np.log(p)) for p in user_probs])
    return novelty


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def _ark(actual, predicted, k=10):
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / len(actual)


# Adapted from https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
def mark(actual, predicted, k=10):
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: int
            The mean average recall at k (mar@k)
    """
    return np.mean([_ark(a, p, k) for a, p in zip(actual, predicted)])


# Precisión clásica
def precision(actual, predicted, k=10):
    return np.nanmean([np.nan if len(p) == 0 else len(np.intersect1d(a, p[:k])) / len(p) for a, p in zip(actual, predicted)])


# Recall clásico
def recall(actual, predicted, k=10):
    return np.nanmean([np.nan if len(a) == 0 else len(np.intersect1d(a, p[:k])) / len(a) for a, p in zip(actual, predicted)])
