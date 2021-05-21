from utils import load_normalize_data, setup_seed, normalize_dense, sample_neighborhood_dataset_tabular
from sklearn.svm import SVR
import numpy as np
import os
import torch
from gime.gime_explainer import Explainer
from collections import Counter
'''
preprocessing dataset and training blac-box models
'''
setup_seed(2020)
dataset = 'wine'
model = 'svr'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = ''.format(dataset)
train_vectors, train_label, valid_vectors, valid_label, test_vectors, test_label, train_mean, train_stddev = load_normalize_data(
    'dataset/{}/{}.csv'.format(dataset, dataset))
train_num, d = train_vectors.shape
valid_num = valid_vectors.shape[0]
test_num = test_vectors.shape[0]
print("train: {} valid: {} test: {}  dim: {}".format(train_num, valid_num,
                                                     test_num, d))
rgr = SVR()
rgr.fit(train_vectors, train_label)
print(np.sqrt(pow(rgr.predict(test_vectors) - test_label, 2).mean()))
'''
sampling neighborhood of training instances
'''
predict_fn = lambda x: rgr.predict(x)

print("Neighborhood not exist, start preparing")
train_neigh_vectors_reshape, train_neigh_preds = sample_neighborhood_dataset_tabular(
    train_vectors, classifier_fn=predict_fn)
valid_neigh_vectors_reshape, valid_neigh_preds = sample_neighborhood_dataset_tabular(
    valid_vectors, classifier_fn=predict_fn)
test_infos = []
for i in range(5):
    test_neigh_vectors_reshape, test_neigh_preds = sample_neighborhood_dataset_tabular(
        test_vectors, classifier_fn=predict_fn)
    test_info = (test_vectors, test_neigh_vectors_reshape, test_neigh_preds)
    test_infos.append(test_info)
train_info = (train_vectors, train_neigh_vectors_reshape, train_neigh_preds)
valid_info = (valid_vectors, valid_neigh_vectors_reshape, valid_neigh_preds)

explainer = Explainer(
    max_iter=14,
    group_num=10,
    selected_feature_num=5,
    feature_num=d,
    device=device)
exp = explainer.fit(train_info, valid_info)

from collections import defaultdict
rmse_scores = []
for r in range(5):
    test_vectors, test_neigh_vectors_reshape, test_neigh_preds = test_infos[r]
    test_vectors = normalize_dense(test_vectors)
    tensor_test_vectors = torch.Tensor(test_vectors).to(device)
    explainer.groupwise_centers.eval()

    weights = explainer.groupwise_centers(
        tensor_test_vectors).detach().cpu().numpy()  # S*R
    test_cluster_ids = weights.argmax(axis=1)
    selected_weights = weights.max(axis=1)

    test_explain_losses = []  # R*S
    test_sample_num = test_vectors.shape[0]
    test_neigh_preds_reshape = test_neigh_preds.reshape(-1)  # S*P
    for i in range(explainer.group_num):
        test_approximate = explainer.groupwise_exps[i].predict(
            test_neigh_vectors_reshape)
        test_explain_loss = pow(test_approximate - test_neigh_preds_reshape,
                                2).reshape(test_sample_num,
                                           -1).mean(axis=1)  # S
        test_explain_losses.append(test_explain_loss)
    test_explain_losses = np.array(test_explain_losses)

    test_real_explain_loss = test_explain_losses[test_cluster_ids,
                                                 list(range(test_sample_num))]

    test_rmse = np.sqrt(test_real_explain_loss.mean())
    print(test_rmse)
    rmse_scores.append(test_rmse)

print("Average Rmse {}+-{} in Five Runs".format(
    np.round(np.mean(rmse_scores), 4), np.round(np.std(rmse_scores), 4)))
