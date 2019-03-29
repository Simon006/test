"""
Likelihood Lab
XingYu
"""


import numpy as np
import random as rd
import matplotlib.pyplot as plt
from math import log
from collections import Counter
from sklearn.datasets import load_wine


class Node:
    def __init__(self):
        # node information
        self.depth = 0
        self.split_index = None
        self.split_value = None
        self.category = None
        self.is_terminal = False

        # children nodes
        self.left_node = None
        self.right_node = None


class DecisionTree:
    def __init__(self, input_dim, maximal_depth=1000, minimal_samples=1, criterion='gini'):
        # classifier information
        self._input_dim = input_dim
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples
        self._criterion = criterion

        # define the tree root
        self._root = Node()

    def train(self, x, y):
        # check dimensionality
        if len(x[0]) != self._input_dim:
            raise ValueError('input dimension does not match.')

        # construct the tree recursively
        self._build_tree(self._root, x, y)

    def predict(self, x):
        # check dimensionality
        if len(x[0]) != self._input_dim:
            raise ValueError('input dimension does not match.')

        y_predict = []
        for sample in x:
            current_node = self._root
            # classification by iterative bisection
            while True:
                if current_node.is_terminal:
                    y_predict.append(current_node.category)  # classify the sample by the terminal point it encounters
                    break
                elif sample[current_node.split_index] < current_node.split_value:
                    current_node = current_node.left_node
                else:
                    current_node = current_node.right_node
        return np.array(y_predict)

    def evaluate(self, x, y):
        # check dimensionality
        if len(x[0]) != self._input_dim:
            raise ValueError('input dimension does not match.')

        y_predict = self.predict(x)
        correct_num = 0
        for i in range(len(y)):
            if y[i] == y_predict[i]:
                correct_num += 1
            else:
                continue
        accuracy = correct_num / len(y)
        return accuracy

    def _build_tree(self, node, x, y):
        condition1 = len(x) > self._minimal_samples  # enough training data
        condition2 = node.depth < self._maximal_depth  # avoid a decision tree that is too complicated

        if condition1 and condition2:
            best_split_index, best_split_value, x_left, y_left, x_right, y_right = self._split(x, y)
            if (len(x_left) == 0) or (len(x_right) == 0):
                node.split_index = None
                node.split_value = None
                node.is_terminal = True  # create leaf node
                node.category = max(list(y), key=list(y).count)  # majority voting
                node.left_node = None
                node.right_node = None
                return 0
            else:
                node.split_index = best_split_index
                node.split_value = best_split_value
                node.is_terminal = False
                node.category = None
                node.left_node = Node()
                node.right_node = Node()
                node.left_node.depth = node.depth + 1
                node.right_node.depth = node.depth + 1
                self._build_tree(node.left_node, x_left, y_left)  # recursion
                self._build_tree(node.right_node, x_right, y_right)  # recursion
        else:
            node.split_index = None
            node.split_value = None
            node.is_terminal = True  # create leaf node
            node.category = max(list(y), key=list(y).count)  # majority voting
            node.left_node = None
            node.right_node = None
            return 0

    def _split(self, x, y):
        best_split_index = None
        best_split_value = None
        best_x_left = None
        best_y_left = None
        best_x_right = None
        best_y_right = None
        best_classification_performance = 100000000

        for i in range(self._input_dim):
            for sample in x:
                if self._criterion == 'gini':
                    classification_performance, x_left, y_left, x_right, y_right = self._gini(i, sample[i], x, y)
                elif self._criterion == 'entropy':
                    classification_performance, x_left, y_left, x_right, y_right = self._entropy(i, sample[i], x, y)
                else:
                    raise ValueError('self._criterion cannot be ' + self._criterion)

                if classification_performance < best_classification_performance:
                    best_classification_performance = classification_performance
                    best_split_index = i
                    best_split_value = sample[i]
                    best_x_left = x_left
                    best_y_left = y_left
                    best_x_right = x_right
                    best_y_right = y_right

        return best_split_index, best_split_value, best_x_left, best_y_left, best_x_right, best_y_right

    def _gini(self, split_index, split_value, x, y):
        if len(x[0]) != self._input_dim:
            raise ValueError('input dimension does not match.')

        # divide the training data into two groups by index and value
        left_x_list = []
        left_y_list = []
        right_x_list = []
        right_y_list = []
        for i in range(len(x)):
            sample = x[i]
            label = y[i]
            if sample[split_index] < split_value:
                left_x_list.append(sample)
                left_y_list.append(label)
            else:
                right_x_list.append(sample)
                right_y_list.append(label)

        # calculate gini of the left node
        # the key of the dictionary(left_y_stat_dict) is class label;
        # the value of dictionary is the corresponding frequency;
        left_y_stat_dict = Counter(left_y_list)
        gini_value_left = sum([(left_y_stat_dict[key]/len(left_y_list))*(1-(left_y_stat_dict[key]/len(left_y_list)))
                               for key in left_y_stat_dict])
        left_prob = len(left_y_list) / (len(left_y_list) + len(right_y_list))

        # calculate gini of the right node
        right_y_stat_dict = Counter(right_y_list)
        gini_value_right = sum([(right_y_stat_dict[key]/len(right_y_list))*(1-(right_y_stat_dict[key]/len(right_y_list)))
                                for key in right_y_stat_dict])
        right_prob = len(right_y_list) / (len(left_y_list) + len(right_y_list))

        # calculate comprehensive gini
        gini_value = gini_value_left * left_prob + gini_value_right * right_prob

        return gini_value, np.array(left_x_list), np.array(left_y_list), np.array(right_x_list), np.array(right_y_list)

    def _entropy(self, split_index, split_value, x, y):
        if len(x[0]) != self._input_dim:
            raise ValueError('input dimension does not match.')

        # divide the training data into two groups by index and value
        left_x_list = []
        left_y_list = []
        right_x_list = []
        right_y_list = []
        for i in range(len(x)):
            sample = x[i]
            label = y[i]
            if sample[split_index] < split_value:
                left_x_list.append(sample)
                left_y_list.append(label)
            else:
                right_x_list.append(sample)
                right_y_list.append(label)

        # calculate entropy of the left node
        # the key of the dictionary(left_y_stat_dict) is class label;
        # the value of dictionary is the corresponding frequency;
        left_y_stat_dict = Counter(left_y_list)
        entropy_value_left = 0
        for key in left_y_stat_dict:
            p = left_y_stat_dict[key] / len(left_y_list)
            if p != 0:
                entropy_value_left -= p * log(p)
            else:
                entropy_value_left -= 0
        left_prob = len(left_y_list) / (len(left_y_list) + len(right_y_list))

        # calculate entropy of the right node
        right_y_stat_dict = Counter(right_y_list)
        entropy_value_right = 0
        for key in right_y_stat_dict:
            p = right_y_stat_dict[key] / len(right_y_list)
            if p != 0:
                entropy_value_right -= p * log(p)
            else:
                entropy_value_right -= 0
        right_prob = len(right_y_list) / (len(left_y_list) + len(right_y_list))

        # calculate comprehensive entropy
        entropy = entropy_value_left * left_prob + entropy_value_right * right_prob

        return entropy, np.array(left_x_list), np.array(left_y_list), np.array(right_x_list), np.array(right_y_list)


if __name__ == '__main__':
    # load wine data
    # each sample has 13 features and 3 possible classes
    wine = load_wine()
    wine_x = wine['data']
    wine_y = wine['target']

    # shuffle the data randomly
    random_idx = rd.sample([i for i in range(len(wine_x))], len(wine_x))
    wine_x = wine_x[random_idx]
    wine_y = wine_y[random_idx]

    # split the data into training data set and testing data set
    train_rate = 0.7
    train_num = int(train_rate*len(wine_x))
    train_x = wine_x[:train_num]
    train_y = wine_y[:train_num]
    test_x = wine_x[train_num:]
    test_y = wine_y[train_num:]

    # construct decision tree classifier and compare the performance of different maximal depth
    acc_list = []
    depth_list = [(i+1) for i in range(100)]
    for depth in depth_list:
        dt = DecisionTree(input_dim=13, maximal_depth=depth, minimal_samples=1)
        dt.train(train_x, train_y)
        acc = dt.evaluate(test_x, test_y)
        acc_list.append(acc)
    plt.plot(depth_list, acc_list, marker='^', color='lightgray')
    plt.xlabel('Maximal Depth')
    plt.ylabel('Accuracy')
    plt.title('Performance of different maximal length')
    plt.show()

    # construct decision tree classifier and compare the performance of different minimal samples
    acc_list = []
    sample_num_list = [(i+1) for i in range(100)]
    for sample_num in sample_num_list:
        dt = DecisionTree(input_dim=13, maximal_depth=100, minimal_samples=sample_num, criterion='entropy')
        dt.train(train_x, train_y)
        acc = dt.evaluate(test_x, test_y)
        acc_list.append(acc)
    plt.plot(sample_num_list, acc_list, marker='^', color='lightgray')
    plt.xlabel('Minimal Samples Number')
    plt.ylabel('Accuracy')
    plt.title('Performance of different minimal samples number')
    plt.show()
