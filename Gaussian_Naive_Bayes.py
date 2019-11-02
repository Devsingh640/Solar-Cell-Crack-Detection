import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

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

 
# implementing  basic test train split here.   
print("implementing  basic test train split.")
print("##########################################################################")
print("Data set ready to be for further processing : ")
print("Total number of elements in X-festure : ",len(X))
print("Total number of elements in Y-festure : ",len(Y))
print("Implementing test train split on X_feature and Y_feature.")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
print("Data set splitted.")
print("Length for x_train is : ", len(x_train))
print("Length for x_test is : ", len(x_test))
print("Length for y_train is : ", len(y_train))
print("Length for y_test is : ", len(y_test))
print("Data sets ready for training and prediction.")
print("##########################################################################")
print(" ")
# basic test train split implementation completed here.

# implementing basic Gaussian Naive Bayes partial fit here.
print("implementing basic Gaussian Naive Bayes partial fit.")
print("##########################################################################")
print("clf2 in process please wait for results.")
clf2_pf = GaussianNB()
print("Fitting in process:")
clf2_pf.partial_fit(x_train, y_train, np.unique(y_train))
print("Fitting completed.")
print("Prediction in process:")
y_pred = clf2_pf.predict(x_test)
print("Prediction completed.")
print("Calculating accuracy results.")
accuracy_score_GaussianNB_pf = accuracy_score(y_test, y_pred)
print("Accuracy score : ", accuracy_score_GaussianNB_pf)
print("##########################################################################")
print(" ")
# basic Gaussian Naive Bayes partial fit implementation completed here.

# implementing basic Gaussian Naive Bayes complete fit here.
print("implementing basic Gaussian Naive Bayes complete fit.")
print("##########################################################################")
print("clf3 in process please wait for results.")
clf3_cf = GaussianNB()
print("Fitting in process:")
clf3_cf.partial_fit(x_train, y_train, np.unique(y_train))
print("Fitting completed.")
print("Prediction in process:")
y_pred = clf3_cf.predict(x_test)
print("Prediction completed.")
print("Calculating accuracy results.")
accuracy_score_GaussianNB_cf = accuracy_score(y_test, y_pred)
print("Accuracy score : ", accuracy_score_GaussianNB_cf)
print("##########################################################################")
print(" ")
# basic Gaussian Naive Bayes complete fit implementation completed here. 