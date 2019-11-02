import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

push_list = []
estimators = []
list_for_csv = []
header = [['S.No.', 'RANDOM-STATE', 'SCM ACCURACY-SCORE', 'GAUSSIAN_NB_PF ACCURACY-SCORE', 'GAUSSIAN_NB_CF ACCURACY-SCORE', 'LINEAR REGRESSION ACCURACY-SCORE', 'PCA WITH SVC ACCURACY-SCORE']]

# creating initial csv file.
with open('Compiled_Results\Solar_Pannel_Crack_Detection_Using_Train_Test_Split_Compiled_Results.csv', 'w') as CSV:
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

for seed in range(0, 100):
    
    serial_number = (seed + 1)
    list_for_csv.append(serial_number)
    list_for_csv.append(seed)
    
    # implementing  basic test train split here.   
    print("implementing  basic test train split.")
    print("##########################################################################")
    print("Data set ready to be for further processing : ")
    print("Total number of elements in X-festure : ", len(X))
    print("Total number of elements in Y-festure : ", len(Y))
    print("Implementing test train split on X_feature and Y_feature.")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed)
    print("Data set splitted.")
    print("Length for x_train is : ", len(x_train))
    print("Length for x_test is : ", len(x_test))
    print("Length for y_train is : ", len(y_train))
    print("Length for y_test is : ", len(y_test))
    print("Data sets ready for training and prediction.")
    print("##########################################################################")
    print(" ")
    # basic test train split implementation completed here. 
    
    # implementing basic svm here.
    print("implementing basic svm.")
    print("##########################################################################")
    print("clf1 in process please wait for results.")
    clf1 = svm.SVC(gamma='scale')
    estimators.append(('svm', clf1))
    print("Fitting in process:")
    clf1.fit(x_train, y_train)  
    print("Fitting completed.")
    print("Prediction in process:")
    y_pred = clf1.predict(x_test)
    print("Prediction completed.")
    print("Calculating accuracy results.")
    accuracy_score_svm = accuracy_score(y_test, y_pred)
    list_for_csv.append(str(accuracy_score_svm))
    print("Accuracy score : ", accuracy_score_svm)
    print("##########################################################################")
    print(" ")
    # basic svm implementation completed here.

    # implementing basic Gaussian Naive Bayes partial fit here.
    print("implementing basic Gaussian Naive Bayes partial fit.")
    print("##########################################################################")
    print("clf2 in process please wait for results.")
    clf2_pf = GaussianNB()
    estimators.append(('GaussianNB_pf', clf2_pf))
    print("Fitting in process:")
    clf2_pf.partial_fit(x_train, y_train, np.unique(y_train))
    print("Fitting completed.")
    print("Prediction in process:")
    y_pred = clf2_pf.predict(x_test)
    print("Prediction completed.")
    print("Calculating accuracy results.")
    accuracy_score_GaussianNB_pf = accuracy_score(y_test, y_pred)
    list_for_csv.append(accuracy_score_GaussianNB_pf)
    print("Accuracy score : ", accuracy_score_GaussianNB_pf)
    print("##########################################################################")
    print(" ")
    # basic Gaussian Naive Bayes partial fit implementation completed here.
    
    # implementing basic Gaussian Naive Bayes complete fit here.
    print("implementing basic Gaussian Naive Bayes complete fit.")
    print("##########################################################################")
    print("clf3 in process please wait for results.")
    clf3_cf = GaussianNB()
    estimators.append(('GaussianNB_cf', clf3_cf))
    print("Fitting in process:")
    clf3_cf.partial_fit(x_train, y_train, np.unique(y_train))
    print("Fitting completed.")
    print("Prediction in process:")
    y_pred = clf3_cf.predict(x_test)
    print("Prediction completed.")
    print("Calculating accuracy results.")
    accuracy_score_GaussianNB_cf = accuracy_score(y_test, y_pred)
    list_for_csv.append(str(accuracy_score_GaussianNB_cf))
    print("Accuracy score : ", accuracy_score_GaussianNB_cf)
    print("##########################################################################")
    print(" ")
    # basic Gaussian Naive Bayes complete fit implementation completed here.    
    
    # implementing basic linear regression here.
    print("implementing basic linear regression.")
    print("##########################################################################")
    print("clf4 in process please wait for results.")
    clf4 = LinearRegression() 
    estimators.append(('lr', clf4))
    print("Fitting in process:")
    clf4.fit(x_train, y_train)
    print("Fitting completed.")
    print("Prediction in process:")
    y_pred = clf4.predict(x_test)
    print("Prediction completed.")
    print("Calculating accuracy results.")
    accuracy_score_lr = accuracy_score(y_test, y_pred.round(), normalize = True)
    list_for_csv.append(str(accuracy_score_lr))
    print("Accuracy score : ", accuracy_score_lr)
    print("##########################################################################")
    print(" ")
    # basic linear regression implementation completed here.
    
    '''
    # implementing basic pca with svm here.
    print("implementing basic pca with svm.")
    print("##########################################################################")
    print("pca in process please wait for results.")
    pca = PCA(n_components = 2)
    print("Fitting in process for pca :")
    pca.fit(x_train)  
    print("Fitting completed for pca.")
    x_t_train = pca.fit_transform(x_train)
    x_t_test = pca.transform(x_test)
    clf1 = svm.SVC(gamma='scale')
    print("Fitting in process for svm :")
    clf1.fit(x_train, y_train)
    print("Fitting completed for svm.")
    print("Prediction in process:")
    pred = clf1.predict(x_t_test)
    print("Prediction completed.")
    print("Calculating accuracy results.")
    score = clf1.score(x_t_test, y_test)
    accuracy_score_pca_svc = accuracy_score(y_test, pred)
    list_for_csv.append(str(accuracy_score_pca_svc))
    print("Score : ", score)
    print("Pred label", pred)
    print("Accuracy score : ", accuracy_score_pca_svc)
    print("##########################################################################")
    print(" ")
    # basic pca with svm implementation completed here.
    '''
    push_list.append(list_for_csv)
    
    # recording results in csv file.
    with open('Compiled_Results\Solar_Pannel_Crack_Detection_Using_Train_Test_Split_Compiled_Results.csv', 'a') as CSV:
        writer = csv.writer(CSV)
        writer.writerows(push_list)
    CSV.close()
    # recording ends here.
    
    # clearing list starts here.
    list_for_csv.clear()
    push_list.clear()
    # clearing list ends here.    
    
    