import copy
import numpy as np
from math import exp
import random as rd
from collections import Iterable
from sklearn.datasets import load_breast_cancer


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, neuron_list, activation_list, learning_rate, epoch):
        # basic neural net information
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._neuron_list = neuron_list
        self._activation_list = activation_list
        self._learning_rate = learning_rate
        self._epoch = epoch

        # construct network
        self._network = self._initialize_network()

    def train(self, x, y):
        for e in range(self._epoch):
            square_error_sum = 0
            # minimize the loss function in each training sample by gradient descent
            # the loss function is defined as half of the square error between prediction and label
            for index, sample in enumerate(x):
                input_output_record = []
                # forward propagation
                tensor = np.reshape(sample, newshape=(len(sample), 1))
                for forward_step in range(len(self._network)):
                    input_output_record.append(dict())

                    # conduct linear transformation and record the input and linear output
                    input_output_record[forward_step]['input'] = copy.deepcopy(tensor)
                    tensor = np.dot(self._network[forward_step]['weight'], tensor) + self._network[forward_step]['bias']
                    input_output_record[forward_step]['linear_output'] = copy.deepcopy(tensor)

                    # non-linear transformation
                    tensor = self._activation(index=forward_step, vector=tensor)

                # add square error in this sample
                y_difference = np.reshape(tensor, newshape=(1, len(tensor)))[0] - y[index]
                square_error_sum += np.sqrt(np.dot(y_difference, y_difference))

                # backward propagation
                previous_result = np.reshape(y_difference, (len(y_difference), 1)) * self._d_activation(-1, input_output_record[-1]['linear_output'])
                for backward_step in reversed(range(len(self._network))):
                    # calculate partial derivative
                    partial_ok_wk = np.reshape(input_output_record[backward_step]['input'], (1, len(input_output_record[backward_step]['input'])))
                    partial_ok_ik = self._network[backward_step]['weight']
                    partial_ik_o_k_minus_one = self._d_activation(backward_step-1, input_output_record[backward_step-1]['linear_output'])

                    # Stochastic Gradient Descent (SGD)
                    self._network[backward_step]['weight'] -= self._learning_rate * np.dot(previous_result ,partial_ok_wk)
                    self._network[backward_step]['bias'] -= self._learning_rate * previous_result

                    # renew previous result for later training
                    if backward_step > 0:
                        previous_result = np.transpose(np.dot(np.transpose(previous_result), partial_ok_ik)) * partial_ik_o_k_minus_one

            # print the error in this training epoch
            mse = square_error_sum / len(x)
            print('>>>The MSE of the {0} epoch is {1}'.format(str(e + 1), np.round(mse, decimals=5)))

    def predict(self, x):
        y_predict = []
        for index, sample in enumerate(x):
            tensor = np.reshape(sample, newshape=(len(sample),1))
            # forward propagation
            for index, layer in enumerate(self._network):
                # linear transformation
                tensor = np.dot(layer['weight'], tensor) + layer['bias']

                # non-linear transformation
                tensor = self._activation(index=index, vector=tensor)
            y_predict.append(np.reshape(tensor, newshape=(1, len(tensor)))[0])

        y_predict = np.array(y_predict)
        return y_predict

    def evaluate(self, x, y):
        y_predict = self.predict(x)
        error = 0
        for index, row in enumerate(y_predict):
            y_difference = row - y[index]
            error += np.sqrt(np.dot(y_difference, y_difference))
        average_error = error / len(x)
        return average_error

    def _initialize_network(self):
        # check mistake
        if self._output_dim != self._neuron_list[-1]:
            raise ValueError('output dimensionality does not match the neuron number of the last layer.')

        if len(self._neuron_list) < len(self._activation_list):
            raise ValueError('each layer can only have one activation function.')
        elif len(self._neuron_list) > len(self._activation_list):
            raise ValueError('every layer must be equipped with one activation function.')
        else:
            pass

        # initialize the network's layer
        network = []
        for index, neuron_num in enumerate(self._neuron_list):
            layer = dict()
            # define layer weight and bias
            if index == 0:
                layer['weight'] = np.random.normal(loc=0, scale=0.05, size=(neuron_num, self._input_dim))
            else:
                layer['weight'] = np.random.normal(loc=0, scale=0.05, size=(neuron_num, self._neuron_list[index-1]))
            layer['bias'] = np.random.normal(loc=0, scale=0.05, size=(neuron_num, 1))

            # define the activation function(you have two options: rectified linear unit or sigmoid)
            if self._activation_list[index] not in {'relu', 'sigmoid'}:
                raise ValueError('activation not available.')
            else:
                layer['activation'] = self._activation_list[index]

            network.append(layer)

        return network

    def _activation(self, index, vector):
        if vector.shape[1] != 1:
            raise ValueError('activation function can only be applied to column vector.')

        if self._network[index]['activation'] == 'relu':
            result = _rectified_linear_unit(vector)
        elif self._network[index]['activation'] == 'sigmoid':
            result = _sigmoid(vector)
        else:
            raise ValueError

        return result

    def _d_activation(self, index, vector):
        if vector.shape[1] != 1:
            raise ValueError('activation function can only be applied to column vector.')

        if self._network[index]['activation'] == 'relu':
            result = _derivative_rectified_linear_unit(vector)
        elif self._network[index]['activation'] == 'sigmoid':
            result = _derivative_sigmoid(vector)
        else:
            raise ValueError
        return result


def _sigmoid(vector):
    if vector.shape[1] != 1:
        raise ValueError('activation function can only be applied to column vector.')

    result = np.zeros((len(vector), 1))
    for index, value in enumerate(vector):
        result[index][0] = 1 / (1 + exp(-value[0]))

    return result


def _derivative_sigmoid(vector):
    if vector.shape[1] != 1:
        raise ValueError('activation function can only be applied to column vector.')

    temp = copy.deepcopy(vector)
    result = _sigmoid(temp) * (1 - _sigmoid(temp))
    return result


def _rectified_linear_unit(vector):
    if vector.shape[1] != 1:
        raise ValueError('activation function can only be applied to column vector.')

    result = copy.deepcopy(vector)
    result[result < 0] = 0
    return result


def _derivative_rectified_linear_unit(vector):
    if vector.shape[1] != 1:
        raise ValueError('activation function can only be applied to column vector.')

    result = copy.deepcopy(vector)
    result[result <= 0] = 0
    result[result > 0] = 1
    return result


if __name__ == '__main__':
    # load data
    breast_cancer = load_breast_cancer()
    breast_cancer_x = breast_cancer['data']
    breast_cancer_y = breast_cancer['target']

    # shuffle the data randomly
    random_idx = rd.sample([i for i in range(len(breast_cancer_x))], len(breast_cancer_x))
    breast_cancer_x = breast_cancer_x[random_idx]
    breast_cancer_y = breast_cancer_y[random_idx]
    if isinstance(breast_cancer_y[0], Iterable):
        pass
    else:
        breast_cancer_y = np.reshape(breast_cancer_y, newshape=(len(breast_cancer_y), 1))

    # Normalize the data's column to [0,1]
    for j in range(len(breast_cancer_x[0])):
        min_value = min([breast_cancer_x[i][j] for i in range(len(breast_cancer_x))])
        max_value = max([breast_cancer_x[i][j] for i in range(len(breast_cancer_x))])
        for i in range(len(breast_cancer_x)):
            breast_cancer_x[i][j] = (breast_cancer_x[i][j] - min_value) / (max_value - min_value)

    # split the data into training data set and testing data set
    train_rate = 0.8
    train_num = int(train_rate*len(breast_cancer_x))
    train_x = breast_cancer_x[:train_num]
    train_y = breast_cancer_y[:train_num]
    test_x = breast_cancer_x[train_num:]
    test_y = breast_cancer_y[train_num:]

    # train neural net to predict
    dnn = NeuralNetwork(input_dim=len(train_x[0]), output_dim=len(train_y[0]),
                        neuron_list=[5, 3, 1], activation_list=['relu', 'relu', 'sigmoid'],
                        learning_rate=0.01, epoch=500)
    dnn.train(x=train_x, y=train_y)
    average_square_error = dnn.evaluate(x=test_x, y=test_y)
    print('Mean Square Error On Test: ' + str(average_square_error))
