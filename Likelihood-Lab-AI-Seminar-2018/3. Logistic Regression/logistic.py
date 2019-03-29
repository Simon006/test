"""
Likelihood Lab
XingYu
"""
from sklearn.datasets import load_breast_cancer
from math import exp
import matplotlib.pyplot as plt
import random as rd
import numpy as np


class Logistic:
    def __init__(self, input_size, learning_rate, epoch):
        # Hyper-parameter
        self._weight_num = input_size + 1  # Contains one bias term
        self._learning_rate = learning_rate
        self._epoch = epoch

        # Random Normal Distributed initial weight
        self._weight = np.random.randn(self._weight_num)

    def predict(self, x):
        y_predict = []
        for sample in x:
            result = self._logistic(np.dot(sample, self._weight[1:]) + self._weight[0])
            y_predict.append(result)

        y_predict = np.array(y_predict)
        return y_predict

    def train(self, x, y):
        # Check length error
        if len(x) != len(y):
            raise ValueError('The length of x and y do not match.')

        # Stochastic Gradient Descent
        for e in range(self._epoch):
            for i in range(len(x)):
                gradient = (self.predict(np.array([x[i]]))[0] - y[i]) * np.append(np.array([1]), x[i])
                self._weight = self._weight - self._learning_rate * gradient

    def evaluate(self, x, y):
        # Check length error
        if len(x) != len(y):
            raise ValueError('The length of x and y do not match.')

        # Prediction
        y_prediction = self.predict(x)

        # MSE
        difference = y_prediction - y
        mse = sum([diff**2 for diff in difference]) / len(difference)

        # Accuracy
        y_prediction_class = np.round(y_prediction)
        correct_num = 0
        for i in range(len(y_prediction)):
            if y[i] == y_prediction_class[i]:
                correct_num += 1
            else:
                continue
        accuracy = correct_num / len(y)
        return mse, accuracy

    def feature_importance(self):
        return abs(self._weight)

    def _logistic(self, z):
        return 1 / (1 + exp(-z))


if __name__ == '__main__':
    # Import breast cancer data
    breast_cancer = load_breast_cancer()

    # Separate input(x) and output(y)
    data_x = breast_cancer['data']
    data_y = breast_cancer['target']

    # Normalize the data's column to [0,1]
    for j in range(len(data_x[0])):
        min_value = min([data_x[i][j] for i in range(len(data_x))])
        max_value = max([data_x[i][j] for i in range(len(data_x))])
        for i in range(len(data_x)):
            data_x[i][j] = (data_x[i][j] - min_value) / (max_value - min_value)

    # Shuffle the data
    random_idx = rd.sample([i for i in range(len(data_x))], len(data_x))
    data_x = data_x[random_idx]
    data_y = data_y[random_idx]

    # Separate training and testing data set
    train_rate = 0.1
    sample_num = len(data_x)
    train_sample_num = int(train_rate * sample_num)
    train_x = data_x[:train_sample_num]
    train_y = data_y[:train_sample_num]
    test_x = data_x[train_sample_num:]
    test_y = data_y[train_sample_num:]

    # See the effect of epoch parameter
    mse_list = []
    acc_list = []
    epoch_list = [i+1 for i in range(200)]
    for epoch in epoch_list:
        lg = Logistic(len(data_x[0]), 0.01, epoch)
        lg.train(train_x, train_y)
        mse, acc = lg.evaluate(test_x, test_y)
        mse_list.append(mse)
        acc_list.append(acc)

    plt.plot(epoch_list, mse_list, color='lightgray')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

    plt.plot(epoch_list, acc_list, color='lightgray')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # See the effect of learning rate parameter
    mse_list = []
    acc_list = []
    lr_array = np.linspace(0.05, 25.0, 300)
    for lr in lr_array:
        lg = Logistic(len(data_x[0]), lr, 1)
        lg.train(train_x, train_y)
        mse, acc = lg.evaluate(test_x, test_y)
        mse_list.append(mse)
        acc_list.append(acc)

    plt.plot(lr_array, mse_list, color='lightgray')
    plt.xlabel('LearningRate')
    plt.ylabel('MSE')
    plt.show()

    plt.plot(lr_array, acc_list, color='lightgray')
    plt.xlabel('LearningRate')
    plt.ylabel('Accuracy')
    plt.show()

    # See feature importance
    lg = Logistic(len(data_x[0]), 0.1, 20)
    lg.train(train_x, train_y)
    plt.plot(lg.feature_importance(), color='lightgray')
    plt.xlabel('FeatureNumber')
    plt.ylabel('Importance')
    plt.show()

    # Plot the logistic function
    x = np.linspace(-5, 5, 10000)
    y = np.array([1/(1+exp(-value)) for value in x])
    plt.plot(x, y, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
