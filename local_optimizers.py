from TreeStructures import TreeNode, ClassificationTree
import numpy as np

class TAO:
    def __init__(self, tree, oblique = False, train_on_all_features = True):
        self.classification_tree = tree
        self.X = []
        self.y = []
        self.train_on_all_features = train_on_all_features

    def evolve(self, X, y, n_iter = 7, min_size_prune = 1):
        self.X = X
        self.y = y
        for i in range(n_iter):
            #print("TAO iter ", i, " di ", n_iter)
            for depth in reversed(range(self.classification_tree.depth + 1)):
                #print("Ottimizzo depth", depth, "....")
                T = self.classification_tree
                nodes = ClassificationTree.get_nodes_at_depth(depth, T)
                #print ([node.id for node in nodes])

                for node in nodes:
                    self.optimize_nodes(node)

                #Rimetto apposto i punti associati ad ogni nodo
                self.classification_tree.build_idxs_of_subtree(X, range(len(X)), T.tree[0], oblique = T.oblique)
        #ClassificationTree.restore_tree(self.classification_tree)
        self.prune()


    def optimize_nodes(self, node):
            #print ("node id = ", node.id, "depth = ", node.depth)
            #print("ottimizzo nodo: ", node.id)
            T = self.classification_tree
            X = self.X
            y = self.y
            if node.data_idxs:
                if node.is_leaf :
                    #Se il nodo è una foglia ottimizzo semplicemente settando la predizione
                    #alla classe più comune tra le instanze che arrivano a quella foglia
                    #print(max(node.data_idxs))
                    best_class = np.bincount(y[node.data_idxs]).argmax()
                    node.value = best_class
                    care_points_idxs = node.data_idxs
                    #print("Node value: ", node.value)
                    #print("Best class: ", best_class)
                else:
                    #In questo caso devo risolvere un problema di minimizzazione
                    #I punti da tenere in considerazione sono solo quelli tali che la
                    #predizione dei due figli del nodo corrente è diversa, e in almeno
                    #uno il punto viene classificato correttamente
                    care_points_idxs = []
                    correct_classification_tuples = {}
                    for data_idx in node.data_idxs:
                        left_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.left_node, T.oblique)
                        right_label_predicted = T.predict_label(X[data_idx].reshape((1, -1)), node.right_node, T.oblique)
                        correct_label = y[data_idx]
                        if (left_label_predicted != right_label_predicted and right_label_predicted==correct_label):
                            correct_classification_tuples[data_idx] = 1
                            care_points_idxs.append(data_idx)
                        elif (left_label_predicted != right_label_predicted and left_label_predicted==correct_label):
                            care_points_idxs.append(data_idx)
                            correct_classification_tuples[data_idx] = 0
                    self.TAO_best_split(node.id, X, y, care_points_idxs, correct_classification_tuples, T)
                    #print("care_points: ", care_points_idxs)

                #print ("indici nodo prima", node.id, "--->", node.data_idxs)
                #prec = node.data_idxs
                #print (set(prec) == set(node.data_idxs))


    def TAO_best_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):

        #Itero sulle componenti
        #print ("Ottimizzo nodo:", node_id)
        if len(care_points_idxs) > 1:
            #print(len(care_points_idxs))
            if T.oblique:
                best_t, best_weights = self.TAO_best_SVM_split(node_id, X, y, care_points_idxs, correct_classification_tuples, T)
                T.tree[node_id].intercept = best_t
                T.tree[node_id].weights = best_weights
            else:
                best_t, best_feature = self.TAO_best_parallel_split(node_id, X, y, care_points_idxs, correct_classification_tuples, T)
                #print("finale: ", error_best)
                T.tree[node_id].threshold = best_t
                T.tree[node_id].feature = best_feature


    #Qui faccio uno split parallelo agli assi
    def TAO_best_parallel_split(self, node_id, X, y, care_points_idxs, correct_classification_tuples, T):
        #print ("node id:", node_id)
        error_best = np.inf
        best_t = T.tree[node_id].threshold
        best_feature = T.tree[node_id].feature
        if self.train_on_all_features:
            features = range(len(X[0]))
        else:
            features = ClassificationTree.get_features(T)

        for j in features:
            #print(j)
            #Prendo tutte le j-esime componenti del dataset e le ordino
            #in modo crescente
            vals = {}
            for point_idx in care_points_idxs:
                vals[point_idx] = X[point_idx, j]

            values = sorted(vals.items(), key=lambda x: x[1])
            sorted_indexes = [tuple[0] for tuple in values]
            #plt.scatter(X[sorted_indexes, j], range(len(values)), s=0.4, c=list(correct_classification_tuples.values()))
            #plt.show()
            T.tree[node_id].feature = j
            thresh = 0.5*(X[sorted_indexes[0], j]+X[sorted_indexes[1], j])
            actual_loss, start = self.zero_one_loss(node_id, X, y, sorted_indexes, correct_classification_tuples, thresh, sorted_indexes)

            #if j==2:
                #base = actual_loss
            #actual_loss = self.binary_loss(node_id, X, y, sorted_indexes[i], correct_classification_tuples[sorted_indexes[i]], actual_loss, thresh)
            #print ("loss: ", actual_loss, "n punti: ", len(care_points_idxs))
            #print("vecchia: ", self.vecchia_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh))
            #Ciclo su ogni valore di quella componente e provo tutti gli split
            #possibili
            if actual_loss < error_best:
                error_best = actual_loss
                best_t = thresh
                best_feature = j
            i = start
            while i < len(sorted_indexes)-1:


                thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])

                #sum, k = self.binary_loss(X, j, sorted_indexes, i, correct_classification_tuples)
                #actual_loss += sum


                #Qualche print per debug
                '''
                if j==2 and node_id==3:
                    ones = [correct_classification_tuples[k] for k in sorted_indexes]
                    print("base: ", base, "  n punti:", len(sorted_indexes), "    loss:", actual_loss, "     thresh:", thresh, "     x:", X[sorted_indexes[i], j], "    prediction: ", correct_classification_tuples[sorted_indexes[i]])
                    print(X[sorted_indexes, j])
                    print()
                '''
                #actual_loss = self.zero_one_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, thresh)

                s, k = self.binary_loss(X, j, sorted_indexes, i, correct_classification_tuples)
                #print ("dopo binary")
                actual_loss += s
                if actual_loss < error_best:
                    error_best = actual_loss
                    best_t = thresh
                    best_feature = j
                i+=k

            '''
            #print(error_best)
            #errors = [self.binary_loss(node_id, X, y, care_points_idxs, correct_classification_tuples, threshes[i]) for i in range(len(values)-1)]
            #min_index = np.argmin(errors)
        '''
        #Ancora print per debug
        #if node_id==1:
            #print("finale: ", error_best)
            #print ("ones:", ones, " len:  ", len(ones), "   care_p_len:", len(care_points_idxs))
            #print("best_feature:", best_feature, "   best_thresh:", best_t)



        return best_t, best_feature


    #Alla fine tolgo dead branches e pure subtrees
    def prune(self, min_size=1):
        #Prima controllo se ci sono sottoalberi puri.
        #Visito l'albero e verifico se esistono nodi branch ai quali arrivano punti associati a solo una label
        #Poi vedo se il nodo attuale è morto
        T = self.classification_tree
        stack = [T.tree[0]]
        while (stack):
            actual = stack.pop()
            if len(actual.data_idxs) > 0:
                #Se il nodo è puro
                if not actual.is_leaf and all(i == self.y[actual.data_idxs[0]] for i in self.y[actual.data_idxs]):
                    #Devo far diventare una foglia questo nodo con valore pari alla label
                    actual.is_leaf = True
                    actual.left_node = None
                    actual.right_node = None
                    actual.left_node_id = -1
                    actual.right_node_id = -1
                    actual.value = self.y[actual.data_idxs[0]]

                #Se il nodo ha un figlio morto devo sostituire il padre con l'altro figlio
                elif not actual.is_leaf and len(actual.left_node.data_idxs) < min_size:
                    stack.append(actual.right_node)
                    ClassificationTree.replace_node(actual, actual.right_node, T)
                elif not actual.is_leaf and len(actual.right_node.data_idxs) < min_size:
                    stack.append(actual.left_node)
                    ClassificationTree.replace_node(actual, actual.left_node, T)
                elif not actual.is_leaf:
                    stack.append(actual.right_node)
                    stack.append(actual.left_node)
            ClassificationTree.restore_tree(T)






    #Definico la loss base del nodo.
    #Significa che parto considerando come thresh la metà del minimo valore su quella componente.
    #Questo mi consente di evitare un O(n) e quindi
    #di ciclare su ogni punto del dataset per calcolare di nuovo la loss cambiando il
    #thresh. Ad ogni cambio, essendo i dati ordinati per componente la loss cambia
    #al più di una unità +/- 1. OCCORRE FARE ATTENZIONE A PUNTI UGUALI! VEDI BINARY LOSS
    def zero_one_loss(self, node_id, X, y, care_points_idxs, correct_classification_tuples, thresh, sorted_indexes):
        loss = 0
        T = self.classification_tree.tree
        node = T[node_id]
        #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        targets = [correct_classification_tuples[x] for x in care_points_idxs]
        data = X[care_points_idxs]
        predictions = np.array([0 if sample[node.feature] <= thresh else 1 for sample in data[:,]])
        n_misclassified = np.count_nonzero(targets-predictions)
        k=0
        i=0
        #print("prima del while")
        while k+i< len(sorted_indexes) and X[sorted_indexes[k+i], node.feature] == X[sorted_indexes[i], node.feature]:
            k+=1
        return n_misclassified, k
        #return loss


    #Definisco la loss in eq (4) per il problema al singolo nodo
    def binary_loss(self, X, feature, sorted_indexes, i, targets):

        #Dato un punto del nodo verifico dove questo lo inoltrerebbe
        #se il punto viene inoltrato su un figlio che porta a misclassificazione
        #è errore
        #print("loss")
        #sum = 0
        #k = 1
        #if targets[sorted_indexes[i]] == 0:
            #sum -= 1
        #else:
            #sum +=1

        #Vedo se ci sono punti uguali che potrebbero dar fastidio con la loss.
        #Se mi fermassi a vedere solo il primo potrei ottenere che lo split migliore
        #sia uno tale che dopo ci sono altri punti uguali che però sono della classe sbagliata
        #se dopo provassi a splittare su quei punti la loss ovviamente si alzerebbe ma
        #nel frattempo, essendo uguali, mi sarei salvato come miglior split proprio il primo.
        #Occorre decidere se conviene eseguire lo split guardando insieme tutti i punti uguali
        #costruendo una loss per maggioranza
        #devo quindi pesare la loss su quanti punti sono classificati bene tra quelli uguali
        #dunque devo guardare i consecutivi
        sum = 0
        k = 0
        while k+i < len(sorted_indexes) and X[sorted_indexes[k+i], feature] == X[sorted_indexes[i], feature]:
            if targets[sorted_indexes[k+i]] == 0:
                sum -= 1
            else:
                sum +=1
            k+=1
        #print (k)
        return sum, k
