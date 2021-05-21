from functools import partial
import os
import pickle
import warnings
import sklearn
import torch
from sklearn.utils import check_random_state
import numpy as np
import random
import scipy as sp
from scipy.sparse import csr_matrix, vstack
import pandas as pd
from sklearn.model_selection import train_test_split


def sample_neighborhood_tabular(target,
                                classifier_fn,
                                perturb_num=10,
                                scale=0.1,
                                protected=[]):
    data = []
    d = target.shape[0]
    for j in range(perturb_num):
        noise = np.random.normal(loc=0.0, scale=1.0, size=d)
        if len(protected) != 0:
            noise[protected] = 0
        pert = target + scale * noise
        if len(protected) != 0:
            if np.random.rand() > 0.85:
                target_id = np.random.choice(protected)
                pert[protected] = 0
                pert[target_id] = 1
        data.append(pert)
    labels = classifier_fn(data)
    return data, labels


def sample_neighborhood_dataset_tabular(dataset,
                                        classifier_fn,
                                        perturb_num=10,
                                        protected=[]):
    neigh_vectors = []
    neigh_preds = []
    cnt = 0
    for data in dataset:
        cnt += 1
        if cnt % 500 == 0:
            print('finish sampling neighborhood for {} instances'.format(cnt))
        data, labels = sample_neighborhood_tabular(data,
                                                   classifier_fn,
                                                   perturb_num,
                                                   protected=protected)
        neigh_vectors.append(data)
        neigh_preds.append(labels)
    neigh_preds = np.array(neigh_preds)
    return np.vstack(neigh_vectors), neigh_preds


def sample_neighborhood(target,
                        classifier_fn,
                        vectorizer,
                        perturb_num=10,
                        kernel_width=25,
                        distance_metric='cosine',
                        label=[1],
                        random_seed=2020):
    def distance_fn(x):
        return sklearn.metrics.pairwise.pairwise_distances(
            x, x[0], metric=distance_metric).ravel() * 100

    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d**2) / kernel_width**2))

    kernel_fn = partial(kernel, kernel_width=kernel_width)
    ini_words = target.split(' ')
    words = list(set(ini_words))
    doc_size = len(words)
    indexed_dict = dict([(words[i], i) for i in range(doc_size)])

    random_state = check_random_state(random_seed)
    sample = random_state.randint(1, doc_size + 1, perturb_num - 1)
    data = np.ones((perturb_num, doc_size))
    data[0] = np.ones(doc_size)
    features_range = range(doc_size)
    inverse_data = [target]
    for i, size in enumerate(sample, start=1):
        inactive = random_state.choice(features_range, size, replace=False)
        inactive_indices = [indexed_dict[words[j]] for j in inactive]
        data[i, inactive] = 0
        perturb_data = ' '.join([
            word for word in ini_words
            if indexed_dict[word] not in inactive_indices
        ])
        inverse_data.append(perturb_data)
    labels = classifier_fn(inverse_data)
    distances = distance_fn(sp.sparse.csr_matrix(data))
    weights = kernel_fn(distances)
    return inverse_data, labels[:, label], weights


def sample_neighborhood_dataset(dataset,
                                classifier_fn,
                                vectorizer,
                                perturb_num=10,
                                kernel_width=25,
                                distance_metric='cosine',
                                label=[1],
                                random_seed=2020):
    neigh_vectors = []
    neigh_preds = []
    neigh_weights = []
    cnt = 0
    for data in dataset:
        cnt += 1
        if cnt % 500 == 0:
            print('finish sampling neighborhood for {} instances'.format(cnt))
        inverse_data, labels, weights = sample_neighborhood(
            data, classifier_fn, vectorizer, perturb_num, kernel_width,
            distance_metric, label, random_seed)
        vectors = vectorizer.transform(inverse_data)
        neigh_vectors.append(vectors)
        neigh_preds.append(labels[:, 0])
        neigh_weights.append(weights)


#     neigh_vectors = np.array(neigh_vectors)
    neigh_preds = np.array(neigh_preds)
    neigh_weights = np.array(neigh_weights)
    return vstack(neigh_vectors), neigh_preds, neigh_weights


def normalize(X):
    X = X.A
    X = (X.T / np.sqrt(pow(X, 2).sum(axis=1))).T
    X = csr_matrix(X)
    return X


def normalize_dense(X):
    X = (X.T / np.sqrt(pow(X, 2).sum(axis=1))).T
    return X


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_normalize_data(source, thresh=.0000000001, protected=[]):
    df_train = pd.read_csv(source, header=None).dropna()

    # Split train, test, valid - Change up train valid test every iteration
    df_train, df_test = train_test_split(df_train, test_size=0.3)
    df_valid, df_test = train_test_split(df_test, test_size=0.5)

    # delete features for which all entries are equal (or below a given threshold)
    train_stddev = df_train[df_train.columns[:]].std()
    drop_small = np.where(train_stddev < thresh)
    if train_stddev[df_train.shape[1] - 1] < thresh:
        print("ERROR: Near constant predicted value")
    df_train = df_train.drop(drop_small[0], axis=1)
    df_test = df_test.drop(drop_small[0], axis=1)
    df_valid = df_valid.drop(drop_small[0], axis=1)

    # Calculate std dev and mean
    train_stddev = df_train[df_train.columns[:]].std()
    train_mean = df_train[df_train.columns[:]].mean()

    # protect certain columns
    if len(protected) > 0:
        print(train_stddev)
        train_stddev[protected] = 1
        train_mean[protected] = 0

    # Normalize to have mean 0 and variance 1
    df_train1 = (df_train - train_mean) / train_stddev
    df_valid1 = (df_valid - train_mean) / train_stddev
    df_test1 = (df_test - train_mean) / train_stddev

    # Convert to np arrays
    X_train = df_train1[df_train1.columns[1:-1]].values
    y_train = df_train1[df_train1.columns[-1]].values

    X_valid = df_valid1[df_valid1.columns[1:-1]].values
    y_valid = df_valid1[df_valid1.columns[-1]].values

    X_test = df_test1[df_test1.columns[1:-1]].values
    y_test = df_test1[df_test1.columns[-1]].values

    return X_train, y_train, X_valid, y_valid, X_test, y_test, np.array(
        train_mean), np.array(train_stddev)


#get LIME's coefficients for a particular point
# This num_samples is the default parameter from LIME's github implementation of explain_instance
def unpack_coefs(explainer, x, predict_fn, num_features, num_samples=5000):
    d = x.shape[0]
    coefs = np.zeros((d))

    exp = explainer.explain_instance(x,
                                     predict_fn,
                                     num_features=num_features,
                                     num_samples=num_samples)

    coef_pairs = exp.local_exp[1]
    for pair in coef_pairs:
        coefs[pair[0]] = pair[1]

    intercept = exp.intercept[1]

    return coefs, intercept
