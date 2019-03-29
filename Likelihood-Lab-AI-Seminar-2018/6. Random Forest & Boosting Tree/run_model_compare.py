import random as rd
from sklearn.datasets import load_wine
from decision_tree import DecisionTree
from boosting_tree import BoostingTree
from random_forest import RandomForest


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

    # compare the performances of random forest, boosting tree and decision tree
    rf = RandomForest(input_dim=len(train_x[0]), tree_num=100, maximal_depth=2, minimal_samples=10, criterion='gini')
    bt = BoostingTree(input_dim=len(train_x[0]), tree_num=100, maximal_depth=2, minimal_samples=10, criterion='gini')
    dt = DecisionTree(input_dim=len(train_x[0]), maximal_depth=10, minimal_samples=5, criterion='gini')
    rf.train(train_x, train_y)
    bt.train(train_x, train_y)
    dt.train(train_x, train_y)
    acc_rf = rf.evaluate(test_x, test_y)
    acc_bt = bt.evaluate(test_x, test_y)
    acc_dt, _, _, _ = dt.evaluate(test_x, test_y)
    print('Random Forest Accuracy: ' + str(acc_rf))
    print('Boosting Tree Accuracy: ' + str(acc_bt))
    print('Decision Tree Accuracy: ' + str(acc_dt))
