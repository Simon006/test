"""
Likelihood Lab
XingYu
"""
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


class Knn:
    def __init__(self, k, x_train, y_train):
        self._k = k
        self._x_train = x_train
        self._y_train = y_train

    def train(self):
        """
        KNN is a lazy learning algorithm.
        Therefore, its train function is void.
        """
        pass

    def predict(self, x_test, y_test):
        # Check length error
        if len(x_test) != len(y_test):
            raise ValueError("length doesn't match")

        y_predict = []
        for sample in x_test:
            # Find K nearest neighbors
            neighbor_list = self._neighbor_search(sample)

            # Majority voting
            predict_result = self._vote(neighbor_list)
            y_predict.append(predict_result)

        # Evaluate
        correct_count = 0
        for i in range(len(y_test)):
            if y_predict[i] == y_test[i]:
                correct_count += 1
            else:
                continue
        acc = correct_count / len(y_test)

        return y_predict, acc

    def _neighbor_search(self, sample):
        # Calculate sample similarity
        distance_list = []
        for sample_train in self._x_train:
            dist = np.linalg.norm(sample_train - sample)
            distance_list.append(dist)

        # Find neighbors
        distance_rank = np.argsort(distance_list)
        k_nearest_neighbors = distance_rank[:self._k]

        return k_nearest_neighbors

    def _vote(self, neighbor):
        # Find candidate target
        target_list = []
        for ind in neighbor:
            target_list.append(self._y_train[ind])

        # Voting
        result = max(target_list, key=target_list.count)

        return result


if __name__ == '__main__':

    # Import digits data
    digits = datasets.load_digits()

    # Data set visualization
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(2, 5, index + 1)
        plt.axis('off')
        plt.imshow(image)
        plt.title('Label: %i' % label)
    plt.show()

    # Get input and output
    image_set = digits['data']  # Contains 1797 (8 by 8) digit images
    target_set = digits['target']  # Contains the corresponding answers to the digits

    # Split train set and test set
    train_rate = 0.5
    sample_num = len(image_set)
    x_train_set = image_set[:int(train_rate * sample_num)]
    y_train_set = target_set[:int(train_rate * sample_num)]
    x_test_set = image_set[int(train_rate * sample_num):]
    y_test_set = target_set[int(train_rate * sample_num):]

    # See the performance of different hyper-parameter k
    acc_list = []
    for neighbor_num in range(1, 30):
        agent = Knn(neighbor_num, x_train_set, y_train_set)
        _, acc = agent.predict(x_test_set, y_test_set)
        acc_list.append(acc)
        print('Prediction Accuracy with ' + str(neighbor_num) + ' neighbors: ' + str(acc))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(acc_list, color='r', label='Accuracy')
    ax.legend(loc=1)
    plt.show()
