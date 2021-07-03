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
from sklearn.tree import _tree

from sklearn.datasets import load_svmlight_file


def get_data(path):
    data = load_svmlight_file(path, dtype=np.float32)
    labels = (data[1] > 0).astype(int)

    return np.asarray(data[0].todense()), labels


if __name__ == '__main__':


    sp, cart = [], []
    sp_depth, cart_depth = [], []
    sp_n_leaves, cart_n_leaves = [], []


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, help="Dataset path light svm format")
    args = parser.parse_args()



    for i in tqdm(range(20)):
        sleep(3)
        X, y = get_data(args.dataset)
        print (X.shape)


        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = i)


        #CART
        clf = tree.DecisionTreeClassifier(random_state=i, min_samples_leaf = 4, max_depth = 4)
        clf = clf.fit(X, y)


        #CART TAO
        #cart_tree = ClassificationTree()
        #cart_tree.initialize_from_CART(X, y, clf)
        #tao = TAO(cart_tree)
        #tao.evolve(X, y)
        #preds = cart_tree.predict_data(X_test, cart_tree.tree[0])
        #cart_score = cart_tree.score(preds, y_test)
        cart_score.append(clf.score(X_test, y_test))
        #cart_depth.append(cart_tree.depth)
        #n_leaves = cart_tree.n_leaves
        #cart_n_leaves.append(n_leaves)
        #tree.plot_tree(clf)
        #c_score = clf.score(X_test, y_test)
        #print(c_score)
        #print("CART test: ", c_score)
        #cart.append(cart_score)

        #Sparse
        root = train_tree(X, y, max_depth = 4, min_samples_leaf = 4)
        sparse_tree = ClassificationTree()
        #sp_depth.append(sparse_tree.get_depth(root))
        sparse_tree.initialize(X, y, root)
        sp_n_leaves.append(sparse_tree.n_leaves)
        #tao = TAO(sparse_tree)
        #tao.evolve(X, y)
        sparse_tree.print_tree_structure()
        preds = sparse_tree.predict_data(X_test, root)
        sparse_score = sparse_tree.score(preds, y_test)
        #print(sparse_score)
        #print("Sparse minimal test: ", sparse_score)
        sp.append(sparse_score)


    print("CART: %s +- %s Mean_depth_cart: %s Mean_leaves_number: %s Sparse: %s +- %s Mean_depth_sparse: %s Mean_leaves_number: %s" % (np.mean(cart), np.std(cart), np.mean(cart_depth), np.mean(cart_n_leaves), np.mean(sp), np.std(sp), np.mean(sp_depth), np.mean(sp_n_leaves)))
