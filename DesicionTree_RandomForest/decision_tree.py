from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        #pass

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        def learnlearn(X,y):
            if entropy(y) == 0: # all the same label -> end of tree
                return {'label':y[0]}
            best_split = {} # split_attr,split_value,left,right
            max_IG = -1 
            current_split = None
            for attribute in range(len(X[0])): # attribute = column indices                  
                unique_value = np.unique([x[attribute] for x in X])
                for value in unique_value:
                    X_left, X_right, y_left, y_right = partition_classes(X, y, attribute, value)
                    IG = information_gain(y, [y_left, y_right])
                    if IG > max_IG:
                        max_IG = IG
                        current_split = [attribute,value]
            if max_IG == 0: # just couldn't split better -> end of tree
                cnt_0_1 = np.bincount(y)
                return {'label': [1, 0][cnt_0_1[0] > cnt_0_1[1]]}
            # record and split
            best_split["split_attr"] = current_split[0]
            best_split["split_value"] = current_split[1]
            X_left, X_right, y_left, y_right = partition_classes(X, y, current_split[0], current_split[1])
            # next level 
            best_split['left'] = learnlearn(X_left, y_left)
            best_split['right'] = learnlearn(X_right, y_right)
            return best_split
    
        self.tree = learnlearn(X,y)

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        the_tree = self.tree
        # trace down the tree
        while 'label' not in the_tree:
            if type(the_tree['split_value']) == str:
                if record[the_tree['split_attr']] == the_tree['split_value']:
                    the_tree = the_tree['left']
                else:
                    the_tree = the_tree['right']
            else:
                if record[the_tree['split_attr']] <= the_tree['split_value']:
                    the_tree = the_tree['left']
                else:
                    the_tree = the_tree['right']
        return the_tree['label']

