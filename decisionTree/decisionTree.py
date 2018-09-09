import sys, os
sys.path.append("../data")
from class_vis import draw
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()


from sklearn.tree import DecisionTreeClassifier as DTC
clf = DTC(min_samples_split=50)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)


#### store your predictions in a list named pred

pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)
print("accuracy = ", acc)
draw(clf, features_test, labels_test, os.getcwd())
