import time
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import torch.optim
import torch
import random
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from collections import Counter

from gime.base_model import base_explainer
from gime.base_model import base_center


def normalize(X):
    X = (X.T / np.sqrt(pow(X, 2).sum(axis=1))).T
    return X


class Explainer:
    def __init__(
        self,
        max_iter,
        group_num,
        selected_feature_num,
        feature_num,
        device,
        epoch=30,
        h_abs=True,
        cut_in=0,
        fit_bias=True,
        c_bias=False,
        ban=[],
        selected_context_num=None,
        random_state=2020,
    ):
        self.max_iter = max_iter
        self.group_num = group_num
        self.selected_feature_num = selected_feature_num
        if selected_context_num is None:
            self.selected_context_num = selected_feature_num
        else:
            self.selected_context_num = selected_context_num
        self.feature_num = feature_num
        self.device = device
        self.epoch = epoch
        self.cut_in = cut_in
        self.h_abs = h_abs
        self.ban = ban

        self.random_state = random_state
        self.fit_bias = fit_bias
        self.c_bias = c_bias
        print('with f_bias: {} h_bias: {} h_abs: {}'.format(
            self.fit_bias, self.c_bias, self.h_abs))

        self.groupwise_exps = [0] * group_num
        self.groupwise_centers = base_center(self.feature_num, self.group_num,
                                             self.c_bias, self.h_abs)

    def initialize(self, X):
        st_time = time.time()
        sample_num, feature_num = X.shape
        for i in range(self.group_num):
            tmp_explainer = base_explainer(feature_num,
                                           self.selected_feature_num,
                                           fit_bias=self.fit_bias)
            self.groupwise_exps[i] = tmp_explainer


#         cluster_ids = np.random.randint(0, self.group_num, size=sample_num)
        kmeans = KMeans(self.group_num, random_state=self.random_state)
        kmeans.fit(X)
        cluster_ids = kmeans.labels_
        init_hs = []
        for i in range(self.group_num):
            inds = np.where(cluster_ids == i)
            init_h = X[inds].mean(axis=0)
            init_hs.append(init_h)
        init_hs = np.array(init_hs)
        self.groupwise_centers.centers.weight.data.copy_(
            torch.Tensor(init_hs).to(self.device))

    def fit(self, train_info, valid_info=None):
        X, neigh_vectors_reshape, neigh_preds = train_info
        X = normalize(X)
        neigh_weights = np.ones_like(neigh_preds)
        neigh_preds_reshape = neigh_preds.reshape(-1)  # S*P
        neigh_weights_reshape = neigh_weights.reshape(-1)  # S*P
        neigh_weights_sum = neigh_weights.sum(axis=1)

        if not (valid_info is None):
            valid_X, valid_neigh_vectors_reshape, valid_neigh_preds = valid_info
            valid_X = normalize(valid_X)
            valid_neigh_weights = np.ones_like(valid_neigh_preds)
            valid_neigh_preds_reshape = valid_neigh_preds.reshape(-1)  # S*P
            valid_neigh_weights_reshape = valid_neigh_weights.reshape(
                -1)  # S*P
            valid_neigh_weights_sum = valid_neigh_weights.sum(axis=1)
            valid_sample_num = valid_X.shape[0]
            best_valid_rmse = 10000
            best_iter = -1

        print("Explaining with GIME!")
        st_time = time.time()
        self.initialize(X)

        self.groupwise_centers.to(self.device)
        tensor_X = torch.Tensor(X).to(self.device)
        if not (valid_info is None):
            valid_tensor_X = torch.Tensor(valid_X).to(self.device)
        sample_num = X.shape[0]
        no_best_iter = 0
        for t in range(self.max_iter):
            self.groupwise_centers.eval()
            weights = self.groupwise_centers(
                tensor_X).detach().cpu().numpy()  # S*R
            cluster_ids = weights.argmax(axis=1)

            #Update Theta
            print("Update Theta")
            for i in range(self.group_num):
                # print("group {}".format(i))
                self.groupwise_exps[i].fit(neigh_vectors_reshape,
                                           neigh_preds_reshape,
                                           sample_weights=(neigh_weights.T *
                                                           weights[:, i]).T)
            explain_losses = []  # R*S

            for i in range(self.group_num):
                approximate = self.groupwise_exps[i].predict(
                    neigh_vectors_reshape)
                explain_loss = (pow(approximate - neigh_preds_reshape, 2) *
                                neigh_weights_reshape).reshape(
                                    sample_num, -1).sum(axis=1)  # S
                explain_losses.append(explain_loss)

            explain_losses = np.array(explain_losses)
            losses = torch.Tensor(explain_losses.T).to(self.device)  # S*R

            self.groupwise_centers.train()
            optimizer = torch.optim.Adam(
                params=self.groupwise_centers.parameters(), lr=1e-3)
            cur_loss = 0
            #Update \Phi stage-1
            print("Update Phi stage-1")
            for i in range(self.epoch):
                optimizer.zero_grad()
                weights = self.groupwise_centers(tensor_X)  # S*R
                g_loss = (weights * losses).sum()
                final_loss = g_loss
                final_loss.backward()
                optimizer.step()
                cur_loss = g_loss.item()

            del optimizer

            #Update \Phi stage-2
            print("Update Phi stage-2")
            if t >= self.cut_in:
                best_mae = np.float('inf')
                no_best_iter = 0
                self.groupwise_centers.sparse(self.selected_context_num,
                                              self.ban)
                masks = torch.Tensor(self.groupwise_centers.masks).to(
                    self.device)
                optimizer = torch.optim.Adam(
                    params=self.groupwise_centers.parameters(), lr=1e-3)
                cur_loss = 0
                for i in range(20 * self.epoch):
                    optimizer.zero_grad()
                    weights = self.groupwise_centers(tensor_X)  # S*R
                    g_loss = (weights * losses).sum()
                    final_loss = g_loss
                    final_loss.backward()
                    for k, v in self.groupwise_centers.named_parameters():
                        if k == 'centers.weight':
                            v.grad *= masks
                    optimizer.step()
                    cur_loss = g_loss.item()

                del optimizer

            self.groupwise_centers.eval()
            weights = self.groupwise_centers(
                tensor_X).detach().cpu().numpy()  # S*R
            cluster_ids = weights.argmax(axis=1)

            real_explain_loss = explain_losses[
                cluster_ids, list(range(sample_num))] / neigh_weights_sum
            print('iter {} rmse {}'.format(t,
                                           np.sqrt(real_explain_loss.mean())))

            self.groupwise_centers.eval()
            if not (valid_info is None):
                valid_explain_losses = []  # R*S
                for i in range(self.group_num):
                    valid_approximate = self.groupwise_exps[i].predict(
                        valid_neigh_vectors_reshape)
                    valid_explain_loss = (
                        pow(valid_approximate - valid_neigh_preds_reshape, 2) *
                        valid_neigh_weights_reshape).reshape(
                            valid_sample_num, -1).sum(axis=1)  # S
                    valid_explain_losses.append(valid_explain_loss)

                valid_explain_losses = np.array(valid_explain_losses)
                valid_weights = self.groupwise_centers(
                    valid_tensor_X).detach().cpu().numpy()  # S*R
                valid_cluster_ids = valid_weights.argmax(axis=1)

                valid_real_explain_loss = valid_explain_losses[
                    valid_cluster_ids,
                    list(range(valid_sample_num))] / valid_neigh_weights_sum
                valid_rmse = np.sqrt(valid_real_explain_loss.mean())
                print('iter {} valid rmse {} tau {}'.format(
                    t, valid_rmse, self.groupwise_centers.tau))
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_iter = t

        print("reach max_iters")
        print("GIME takes {} s".format(time.time() - st_time))
        print("Best valid rmse {} at iter {}".format(best_valid_rmse,
                                                     best_iter))
        return cluster_ids, weights
