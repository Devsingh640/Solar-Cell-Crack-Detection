import numpy as np
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

seed = 7
num_trees = 100
max_features = 7

# loaded contents of  combined data set.
Crack_Detection_ds = np.load("Solar_Pannel_Crack_Detection.npz")
# data set loading ends here.

# printing details about data set starts here.
print(" ")
print("##########################################################################")
print("Shape for X-feature :", Crack_Detection_ds["X"].shape)
print("Type for X-feature : ", type(Crack_Detection_ds["X"]))
X = Crack_Detection_ds["X"]
print("Type for X :", type(X))
print("Shape for X :", X.shape)
print("##########################################################################")
print(" ")
print("##########################################################################")   
print("Shape for Y-feature :", Crack_Detection_ds["Y"].shape)
print("Type for Y-feature : ", type(Crack_Detection_ds["Y"]))
Y = Crack_Detection_ds["Y"]
print("Type for Y :", type(Y))
print("Shape for Y :", Y.shape)
print("##########################################################################")
print(" ")
# printing details about data set ends here.

# implementing  basic k-fold.   
print("implementing  basic k-fold.")
print("##########################################################################")
print("Data set ready to be for further processing : ")
print("Total number of elements in X-festure : ", len(X))
print("Total number of elements in Y-festure : ", len(Y))
print("Implementing test train split on X_feature and Y_feature.")
kfold = model_selection.KFold(n_splits = 10, shuffle = True, random_state = seed)
print("Data set splitted.")
print("Data sets ready for training and prediction.")
print("##########################################################################")
print(" ")
# basic k-fold implementation completed here.

# implementing basic extra trees classifier.
print("implementing basic extra trees classifier.")
print("##########################################################################")
print("clf in process please wait for results.")
clf = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
print("Calculating  results.")
results = model_selection.cross_val_score(clf, X, Y, cv = kfold)
print("Results mean : ", results.mean())
print("##########################################################################")
print(" ")
# basic extra trees classifier implementation completed here.