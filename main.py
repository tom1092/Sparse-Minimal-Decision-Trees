from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from TreeStructures import TreeNode, ClassificationTree
from sklearn.model_selection import train_test_split
from local_optimizers import TAO
from sparse_growing import SparseTreeEstimator
from sklearn import tree
import pandas as pd
import sys
from tqdm.auto import tqdm
from time import sleep
import numpy as np
from sklearn.tree import _tree
import argparse
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_svmlight_file


sp, cart = [], []
sp_depth, cart_depth = [], []
sp_n_leaves, cart_n_leaves = [], []

def get_data(path, categorical = False):

    if categorical:
        d = pd.read_csv(path)
        d = pd.get_dummies(d)
        #print("ciao")
        print(d.describe())
        #data = ""
        #labels = ""
    else:
        data = load_svmlight_file(path, dtype=np.float32)
        labels = (data[1] > 0).astype(int)
        data = np.asarray(data[0].todense())

    return data, labels

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, help="Dataset path light svm format")
    args = parser.parse_args()

    for i in tqdm(range(50)):
        sleep(3)

        #d = pd.read_csv("./datasets/tic-tac-toe_csv.csv")
        #d = pd.get_dummies(d)
        #d = d.replace(to_replace=['o', 'x', 'b'], value=[1.0, 2.0, 3.0])
        #print("ciao")
        #print(d.describe())
        #X = d.drop('class', axis = 1).to_numpy()
        #y = d.get('class').to_numpy()
        #y = np.array([0 if y[i] == False else 1 for i in range(len(y))])


        #data = ""
        #labels = ""
        X, y = get_data(args.dataset, categorical = False)
        #print (X.shape)


        X, X_test, y, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state = i)

        #CART
        clf = tree.DecisionTreeClassifier(random_state=i, max_depth = 4)
        clf = clf.fit(X, y)

        parameters = {'min_samples_leaf' : [2, 4, 6], 'max_depth' : [4, 5]}
        grid = GridSearchCV(clf, parameters, n_jobs = -1)
        #print("prima fit")
        grid.fit(X, y)
        #print("post fit")

        clf = grid.best_estimator_

        #CART TAO
        cart_tree = ClassificationTree()
        cart_tree.initialize_from_CART(X, y, clf)
        #tao = TAO(cart_tree)
        #tao.evolve(X, y)
        preds = cart_tree.predict_data(X_test, cart_tree.tree[0])
        cart_score = cart_tree.score(preds, y_test)
        cart_depth.append(cart_tree.depth)
        n_leaves = cart_tree.n_leaves
        cart_n_leaves.append(n_leaves)
        #tree.plot_tree(clf)
        #c_score = clf.score(X_test, y_test)
        #print(c_score)
        #print("CART test: ", c_score)
        cart.append(cart_score)

        #Sparse



        sparse_tree = SparseTreeEstimator()
        parameters = {'min_samples_leaf' : [2, 4, 6], 'max_depth' : [4, 5]}
        grid = GridSearchCV(sparse_tree, parameters, n_jobs = -1)
        #print("prima fit")
        grid.fit(X, y)
        #print("post fit")
        sparse_tree = grid.best_estimator_
        sp_depth.append(sparse_tree.get_depth())

        sp_n_leaves.append(sparse_tree.get_n_leaves())
        #sparse_tree.print_tree()
        sparse_score = sparse_tree.score(X_test, y_test)
        #print(sparse_score)
        #print("Sparse minimal test: ", sparse_score)
        sp.append(sparse_score)


    print("CART: %s +- %s Mean_depth_cart: %s Mean_leaves_number: %s Sparse: %s +- %s Mean_depth_sparse: %s Mean_leaves_number: %s" % (np.mean(cart), np.std(cart), np.mean(cart_depth), np.mean(cart_n_leaves), np.mean(sp), np.std(sp), np.mean(sp_depth), np.mean(sp_n_leaves)))
