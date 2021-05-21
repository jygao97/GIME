import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.linear_model import Ridge, lars_path, LinearRegression
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances
import heapq
from torch.nn.parameter import Parameter


class base_explainer:
    def __init__(self,
                 all_feature_num,
                 selected_feature_num,
                 fit_bias=True,
                 limit=100):
        self.all_feature_num = all_feature_num
        self.selected_feature_num = selected_feature_num
        self.theta = np.random.rand(all_feature_num)
        self.intercept = np.random.rand()
        self.fit_bias = fit_bias
        self.selected_features = None
        if limit == 100:
            self.limit = all_feature_num
        else:
            self.limit = limit

    def fit(self, X, Y, sample_weights=None):
        reg = LinearRegression(fit_intercept=self.fit_bias)
        if not (sample_weights is None):
            sample_weights = sample_weights.reshape(-1)

        selected_features = self.feature_selection(X, Y,
                                                   self.selected_feature_num,
                                                   self.limit, sample_weights)
        #         print(selected_features)

        reg.fit(X[:, selected_features], Y, sample_weights)
        self.selected_features = selected_features
        self.theta = np.zeros(self.all_feature_num)
        self.theta[selected_features] = reg.coef_
        self.intercept = reg.intercept_
        return

    def predict(self, X):
        linear_results = X.dot(self.theta) + self.intercept
        return linear_results

    def cal_loss(self, X, Y):
        loss = pow(self.predict(X) - Y, 2)
        return loss

    def cal_AE(self, X, Y):
        absolute_error = abs(self.predict(X) - Y)
        return absolute_error

    def feature_selection(self,
                          X,
                          Y,
                          selected_feature_num,
                          limit,
                          sample_weights=None,
                          method='highest_weights'):
        if method == 'highest_weights':
            clf = Ridge(alpha=1.0, random_state=2020, fit_intercept=False)
            clf.fit(X, Y, sample_weights)

            coef = clf.coef_

            feature_weights = sorted(
                [i for i in zip(range(X.shape[1]), coef) if i[0] < limit],
                key=lambda x: np.abs(x[1]),
                reverse=True)
            return np.array(
                [x[0] for x in feature_weights[:selected_feature_num]])
        return None


class base_center(nn.Module):
    """docstring for base_center"""
    def __init__(self, feature_num, group_num, bias, h_abs=True):
        super(base_center, self).__init__()
        self.centers = nn.Linear(feature_num, group_num, bias=bias)
        self.masks = np.ones((group_num, feature_num))

        self.tau = Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.tau, 1)
        self.h_abs = h_abs

        print("feature_num: {}".format(feature_num))

    def forward(self, X):
        center_weights = self.tau * (self.centers.weight.T /
                                     torch.norm(self.centers.weight, dim=1))
        weights = torch.matmul(X, center_weights)
        weights = F.softmax(weights, dim=1)
        return weights

    def sparse(self, selected_feature_num, ban=[]):
        gs = self.centers.weight.cpu().detach().numpy()
        group_num, feature_num = gs.shape
        masks = np.zeros((group_num, feature_num))

        for i in range(gs.shape[0]):
            if self.h_abs:
                if len(ban) != 0:
                    mark = False
                    top_ids = []
                    sorted_ids = sorted([(j, abs(gs[i])[j])
                                         for j in range(feature_num)],
                                        key=lambda x: x[1],
                                        reverse=True)
                    for info in sorted_ids:
                        if info[0] not in ban:
                            top_ids.append(info[0])
                        else:
                            if mark:
                                continue
                            else:
                                top_ids.append(info[0])
                                mark = True
                        if len(top_ids) == selected_feature_num:
                            break
                else:
                    top_ids = heapq.nlargest(selected_feature_num,
                                             range(gs.shape[1]),
                                             abs(gs[i]).take)
            else:
                top_ids = heapq.nlargest(selected_feature_num,
                                         range(gs.shape[1]), gs[i].take)
            masks[i, top_ids] = 1
        gs = gs * masks
        self.masks = masks
        self.centers.weight.data.copy_(torch.tensor(gs))
