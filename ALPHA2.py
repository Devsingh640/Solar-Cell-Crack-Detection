import numpy as np
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

push_list = []
estimators = []
list_for_csv = []
header = [['S.No.', 'RANDOM-STATE', 'EXTRA TREE CLF(MEAN)','RANDOM FOREST CLF(MEAN)', 'BAGGED DECISSION TREE CLF(MEAN)']]

# creating initial csv file.
with open('Compiled_Results\Solar_Pannel_Crack_Detection_Using_Bagging_Algorithms_Compiled_Results.csv', 'w') as CSV:
    writer = csv.writer(CSV)
    writer.writerows(header)
CSV.close()
# creating initial csv file.

# loaded contents of  combined data set.
Crack_Detection_ds = np.load("Solar_Pannel_Crack_Detection.npz")
# data set loading ends here.
  
# printing details about data set starts here.
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

for seed in range(0, 2):
    
    serial_number = (seed + 1)
    list_for_csv.append(serial_number)
    list_for_csv.append(seed)
    
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
    num_trees = 100
    max_features = 7
    print("implementing basic extra trees classifier.")
    print("##########################################################################")
    print("clf1 in process please wait for results.")
    clf1 = ExtraTreesClassifier(n_estimators = num_trees, max_features = max_features)
    estimators.append(('extra trees classifier', clf1))
    print("Calculating  results.") 
    results_etc = model_selection.cross_val_score(clf1, X, Y, cv = kfold)
    list_for_csv.append(str(results_etc.mean()))
    print("Results(MEAN) : ", results_etc.mean())
    print("##########################################################################")
    print(" ")
    # basic extra trees classifier implementation completed here.
    
    # implementing basic random forest  classifier.
    num_trees = 100
    max_features = 7
    print("implementing basic random forest classifier.")
    print("##########################################################################")
    print("clf2 in process please wait for results.")
    clf2 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    estimators.append(('random forest', clf2))
    print("Calculating  results.") 
    results_rfc = model_selection.cross_val_score(clf2, X, Y, cv = kfold)
    list_for_csv.append(str(results_rfc.mean()))
    print("Results(MEAN) : ", results_rfc.mean())
    print("##########################################################################")
    print(" ")
    # basic random forest classifier implementation completed here.
    
    '''
    # implementing basic bagged decission tree classifier.
    cart = DecisionTreeClassifier()
    print("implementing basic bagged decission tree.")
    print("##########################################################################")
    print("clf3 in process please wait for results.")
    clf3 = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    print("Calculating  results.") 
    results_bdt = model_selection.cross_val_score(clf3, X, Y, cv = kfold)
    print("Results(MEAN) : ", results_bdt.mean())
    print("##########################################################################")
    print(" ")
    # basic bagged decission tree implementation completed here.
    '''

    push_list.append(list_for_csv)
    
    # recording results in csv file.
    with open('Compiled_Results\Solar_Pannel_Crack_Detection_Using_Bagging_Algorithms_Compiled_Results', 'a') as CSV:
        writer = csv.writer(CSV)
        writer.writerows(push_list)
    CSV.close()
    # recording ends here.
    
    # clearing list starts here.
    list_for_csv.clear()
    push_list.clear()
    estimators.clear()
    # clearing list ends here.    
    