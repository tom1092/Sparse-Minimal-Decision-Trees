from TreeStructures import TreeNode, ClassificationTree
import numpy as np
from local_optimizers import TAO
from sklearn.base import BaseEstimator


class SparseTreeEstimator(BaseEstimator):

    def __init__(self, min_samples_leaf = 4, min_impurity = 1e-2, max_depth = 4):
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X=None, y=None):
        root = self.train_tree(X, y)
        self.tree = ClassificationTree()
        self.tree.initialize(X, y, root)
        #tao = TAO(self.tree)
        #tao.evolve(X, y)



    def score(self, X, y_true):

        preds = self.predict(X)
        return self.tree.score(preds, y_true)

    def get_n_leaves(self):
        return self.tree.n_leaves


    def get_depth(self):
        return self.tree.get_depth(self.tree.tree[0])

    def print_tree(self):
        self.tree.print_tree_structure()

    def predict(self, X):
        return self.tree.predict_data(X, self.tree.tree[0])


    def restore_impurities(self, node, X, y):
        stack = [node]
        positive = np.count_nonzero(y)
        negative = len(node.data_idxs) - positive
        impurity = 1 -  np.abs(positive - negative)/len(node.data_idxs)
        node.impurity = impurity

        while(stack):
            n = stack.pop()
            if not n.is_leaf:
                indexes_left = np.array([i for i in n.data_idxs if X[i, n.feature] <= n.threshold])
                indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))


                positive_r = np.count_nonzero(y[indexes_right])
                negative_r = len(indexes_right) - positive_r
                #impurity_right = 1 - np.abs(positive_r - negative_r)/len(indexes_right)
                impurity_right = (len(indexes_right) - np.abs(positive_r - negative_r))/len(n.data_idxs)

                positive_l = np.count_nonzero(y[indexes_left])
                negative_l = len(indexes_left) - positive_l
                #impurity_left = 1 - np.abs(positive_l - negative_l)/len(indexes_left)
                impurity_left = (len(indexes_left) - np.abs(positive_l - negative_l))/len(n.data_idxs)

                n.left_node.impurity = impurity_left
                n.right_node.impurity = impurity_right
                stack.append(n.left_node)
                stack.append(n.right_node)

    def get_best_split_purity(self, X, y, data_idxs):

        min_impur = np.inf
        best_f = None
        best_th = None
        alfa = 1
        improve = False
        for feature in range(len(X[0])):
            indices = np.argsort(X[data_idxs, feature])
            subset_X = X[data_idxs[indices], feature]
            subset_y = y[data_idxs[indices]]
            i = 0
            n_left, n_right = 0, len(indices)
            n_positive_left = 0
            n_negative_left = 0
            n_positive_right = np.count_nonzero(subset_y)
            n_negative_right = n_right - n_positive_right

            while (i < len(subset_X) - 1):

                th = (subset_X[i]+subset_X[i+1])/2

                if subset_y[i] == 0:
                    n_negative_left += 1
                    n_negative_right -= 1
                else:
                    n_positive_left += 1
                    n_positive_right -= 1


                k = 1
                while k+i < len(subset_X) and subset_X[k+i] == subset_X[i]:
                    if subset_y[k+i] == 0:
                        n_negative_left += 1
                        n_negative_right -= 1
                    else:
                        n_positive_left += 1
                        n_positive_right -= 1
                    k+=1

                i+=k
                n_left += k
                n_right += -k

                impurity_left = 1-np.abs(n_positive_left - n_negative_left)/len(subset_X)
                impurity_right = 1-np.abs(n_positive_right - n_negative_right)/len(subset_X)

                acc_mean = 1 - (np.abs(n_positive_left-n_negative_left)+np.abs(n_positive_right-n_negative_right))/len(subset_X)
                impurity = alfa*min(impurity_left, impurity_right) + (1-alfa)*acc_mean
                if impurity < min_impur:
                    min_impur = impurity
                    best_f = feature
                    best_th = th
                    improve = True

        if not improve:
            best_th = None
        return best_f, best_th, min_impur



    def train_tree(self, X, y):
        key = 0
        depth = 0

        node = TreeNode(key, depth)
        node.data_idxs = np.array(range(len(X)))
        node.depth = 0
        node.is_leaf = True
        positive = np.count_nonzero(y)
        negative = len(node.data_idxs) - positive
        impurity = 1 -  np.abs(positive - negative)/len(node.data_idxs)
        node.impurity = impurity


        stack = [node]
        i = 0
        while(stack):

            stack.sort(key=lambda x: x.impurity)

            #print(stack)
            n = stack.pop()

            if n.depth == self.max_depth:
                ones = np.count_nonzero(y[n.data_idxs])
                nums = np.array([len(n.data_idxs) - ones, ones])
                n_right.value = np.argmax(nums)
                n.is_leaf = True

            else:

                #SET node attributes to the best
                n.threshold = None
                n.feature, n.threshold , p = self.get_best_split_purity(X, y, np.array(n.data_idxs))

                if n.threshold != None:
                    #print("Node: ", n.id, "Feature", n.feature, "Th: ", n.threshold, "Impurity: ", p)

                    #Get indexes of left and right subset
                    indexes_left = np.array([i for i in n.data_idxs if X[i, n.feature] <= n.threshold])
                    indexes_right = np.array(list(set(n.data_idxs) - set(indexes_left)))


                    if (len(indexes_left) >= self.min_samples_leaf and len(indexes_right) >= self.min_samples_leaf):
                        #Get purity of the two subsets
                        #print(n.feature, n.threshold)
                        #print("si")

                        n.is_leaf = False
                        #Get indexes of left and right subset

                        #print(len(indexes_right))
                        #print("left:", len(indexes_left))
                        #print("right:", len(indexes_right))

                        #Get purity of the two subsets
                        #print(n.feature, n.threshold)
                        #print("si")


                        positive_r = np.count_nonzero(y[indexes_right])
                        negative_r = len(indexes_right) - positive_r
                        #impurity_right = 1 - np.abs(positive_r - negative_r)/len(indexes_right)
                        impurity_right = (len(indexes_right) - np.abs(positive_r - negative_r))/len(n.data_idxs)

                        positive_l = np.count_nonzero(y[indexes_left])
                        negative_l = len(indexes_left) - positive_l
                        #impurity_left = 1 - np.abs(positive_l - negative_l)/len(indexes_left)
                        impurity_left = (len(indexes_left) - np.abs(positive_l - negative_l))/len(n.data_idxs)




                        #Controllo se lo split è valido
                        #if n.impurity - min(impurity_left, impurity_right) >= min_impurity:
                            #Effettuo lo split

                        #Create two children
                        n_left = TreeNode(key+1, depth+1)
                        n_right = TreeNode(key+2, depth+1)
                        n_left.depth = n.depth + 1
                        n_right.depth = n.depth + 1
                        n_left.data_idxs = indexes_left
                        n_right.data_idxs = indexes_right
                        key = key + 2

                        #Aggancio al padre
                        n.left_node = n_left
                        n.right_node= n_right
                        n.left_node_id = n_left.id
                        n.right_node_id = n_right.id


                        ones = np.count_nonzero(y[n_left.data_idxs])
                        nums = np.array([len(n_left.data_idxs) - ones, ones])
                        n_left.value = np.argmax(nums)
                        ones = np.count_nonzero(y[n_right.data_idxs])
                        nums = np.array([len(n_right.data_idxs) - ones, ones])
                        n_right.value = np.argmax(nums)


                        n_left.is_leaf = False
                        n_right.is_leaf = False
                        n_right.impurity = impurity_right
                        n_left.impurity = impurity_left
                        if len(n_left.data_idxs) == 1 or len(n_right.data_idxs) == 1:
                            n_right.is_leaf = True
                            n_left.is_leaf = True
                        else:
                            n_left.is_leaf = True
                            n_right.is_leaf = True
                            stack.append(n_right)
                            stack.append(n_left)


                        '''
                        #Altrimenti il nodo padre non migliora sufficientemente l'impurità, dunque diventa foglia
                        else:
                            ones = np.count_nonzero(y[n.data_idxs])
                            nums = np.array([len(n.data_idxs) - ones, ones])
                            n.value = np.argmax(nums)
                            n.is_leaf = True


                        #Altrimenti controllo la purezza. Qui la foglia sarà il figlio sinistro
                        if  n.impurity - impurity_left < min_impurity and impurity_left < impurity_right:
                            n_left.is_leaf = True
                            #stack.append(n_right)

                        else:
                            n_left.is_leaf = False

                            stack.append(n_left)

                        if n.impurity - impurity_right < min_impurity and impurity_right < impurity_left:
                            n_right.is_leaf = True
                            #print("foglia a destra con ", len(n_right.data_idxs), " punti. Punti a sinistra: ", len(n_left.data_idxs))
                            #stack.append(n_left)
                        else:
                            n_right.is_leaf = False
                            stack.append(n_right)

                        n_right.impurity = impurity_right
                        n_left.impurity = impurity_left
                        '''


                        i = i + 1
                    else:
                        ones = np.count_nonzero(y[n.data_idxs])
                        nums = np.array([len(n.data_idxs) - ones, ones])
                        n.value = np.argmax(nums)
                        n.is_leaf = True

                else:
                    ones = np.count_nonzero(y[n.data_idxs])
                    nums = np.array([len(n.data_idxs) - ones, ones])
                    n.value = np.argmax(nums)
                    n.is_leaf = True

        return node