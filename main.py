from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from TreeStructures import TreeNode, ClassificationTree
from sklearn.model_selection import train_test_split
from local_optimizers import TAO
from sparse_growing import train_tree
from sklearn import tree
import pandas as pd
import sys
from tqdm.auto import tqdm
from time import sleep
import numpy as np


sp, cart = [], []
sp_depth, cart_depth = [], []
for i in tqdm(range(50)):
    sleep(3)
    #Spambase
    #data = pd.read_csv("datasets/spambase.csv")
    #X = data.to_numpy()
    #y = X[:, -1].astype(int)
    #X = X[:, 0:-1]

    #Banknote
    #X = np.load("datasets/banknote_train.npy")
    #y = np.load("datasets/banknote_label.npy")


    #Cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target

    #X = X[y!=2]
    #y = y[y!=2]

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = i)


    #CART
    clf = tree.DecisionTreeClassifier(random_state=i, min_samples_leaf = 1, min_impurity_split = 1e-1)
    clf = clf.fit(X, y)


    #CART TAO
    cart_tree = ClassificationTree()
    cart_tree.initialize_from_CART(X, y, clf)
    #tao = TAO(cart_tree)
    #tao.evolve(X, y)
    preds = cart_tree.predict_data(X_test, cart_tree.tree[0])
    cart_score = cart_tree.score(preds, y_test)
    cart_depth.append(cart_tree.depth)
    #tree.plot_tree(clf)
    #c_score = clf.score(X_test, y_test)
    #print(c_score)
    #print("CART test: ", c_score)
    cart.append(cart_score)

    #Sparse
    root = train_tree(X, y, n_splits = 100, min_points_leaf = 1, min_impurity=1e-1)
    sparse_tree = ClassificationTree()
    sp_depth.append(sparse_tree.get_depth(root))
    sparse_tree.initialize(X, y, root)
    #tao = TAO(sparse_tree)
    #tao.evolve(X, y)
    #sparse_tree.print_tree_structure()
    preds = sparse_tree.predict_data(X_test, root)
    sparse_score = sparse_tree.score(preds, y_test)
    #print(sparse_score)
    #print("Sparse minimal test: ", sparse_score)
    sp.append(sparse_score)


print("CART: %s +- %s Mean_depth_cart: %s Sparse: %s +- %s Mean_depth_sparse: %s" % (np.mean(cart), np.std(cart), np.mean(cart_depth), np.mean(sp), np.std(sp), np.mean(sp_depth)))
