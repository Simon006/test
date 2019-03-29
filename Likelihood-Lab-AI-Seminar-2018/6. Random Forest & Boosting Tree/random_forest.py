import numpy as np
import random as rd
from math import sqrt
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, input_dim, tree_num, maximal_depth, minimal_samples, criterion):
        # basic classifier information
        self._forest_input_dim = input_dim
        self._tree_input_dim = int(sqrt(input_dim))
        self._tree_num = tree_num
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples
        self._criterion = criterion

        # build forest
        self._forest = self._construct_forest()

    def train(self, x, y):
        # each tree is trained independently
        for tree in self._forest:
            # sample the training data with replacement (row sampling)
            sample_index_list = [rd.randrange(len(x)) for i in range(len(x))]
            x_sampled = x[sample_index_list]
            y_sampled = y[sample_index_list]

            # trained on the sampled data(both row sampled and column sampled)
            tree['model'].train(x_sampled[:,tree['feature']], y_sampled)

    def predict(self, x):
        # each tree evaluates the data set independently
        y_vote = np.zeros((self._tree_num, len(x)))
        for index, tree in enumerate(self._forest):
            # make inference on the sampled features
            y_vote[index] = tree['model'].predict(x[:,tree['feature']])

        # majority voting
        y_predicted = np.zeros(len(x))
        for i in range(len(y_predicted)):
            voted_result = list(y_vote[:,i])
            y_predicted[i] = max(list(voted_result), key=list(voted_result).count)

        return y_predicted

    def evaluate(self, x, y):
        y_predict = self.predict(x)
        correct_num = 0
        for i in range(len(y)):
            if y[i] == y_predict[i]:
                correct_num += 1
            else:
                continue
        accuracy = correct_num / len(y)
        return accuracy

    def _construct_forest(self):
        forest = []
        for i in range(self._tree_num):
            tree = dict()
            tree['model'] = DecisionTree(self._tree_input_dim, self._maximal_depth,
                                         self._minimal_samples, self._criterion)
            tree['feature'] = self._feature_bagging()
            forest.append(tree)
        return forest

    def _feature_bagging(self):
        feature_list = []
        while len(feature_list) < self._tree_input_dim:
            index = rd.randrange(self._forest_input_dim)
            if index not in feature_list:
                feature_list.append(index)
            else:
                continue

        feature_mask = np.zeros(self._forest_input_dim)
        for index in feature_list:
            feature_mask[index] = 1
        feature_mask = np.array(feature_mask, dtype=np.bool)

        return feature_mask
