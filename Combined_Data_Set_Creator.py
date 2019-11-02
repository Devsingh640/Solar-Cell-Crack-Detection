import numpy as np

X = []  # empty list that will store final feature images from data set with probability1 and probability 0.
Y = []  # empty list that will contain final probability for each image stored in list X.

# loaded contents of prob0.npy and prob1.npy contents into data_set_prob0 and data_set_prob1 variables respectively.
data_set_prob0 = np.load("Image_with_prpbability-0_1D_np-array.npz")
data_set_prob1 = np.load("Image_with_prpbability-1_1D_np-array.npz")
# data sset loading ends here.

# printing details about data set starts here.
print(" ")
print("##########################################################################")
print("Shape for dataset :", data_set_prob0["arr_0"])
print("Data set list with probbility 0: \n", data_set_prob0["arr_0"])
print("##########################################################################")
print(" ")
print("##########################################################################")   
print("Shape for dataset :", data_set_prob1["arr_0"])
print("Data set list with probbility 1: \n", data_set_prob1["arr_0"])
print("##########################################################################")
print(" ")
# printing details about data set ends here.

# read file contents into variables.
prob0_ds_list = data_set_prob0["arr_0"]
prob1_ds_list = data_set_prob1["arr_0"]
# reading into variable step completes here.

# generated probability np arrays that will be mapped with X-featue.
prob_0_y = np.zeros(len(prob0_ds_list))
prob_1_y = np.ones(len(prob1_ds_list))
# generation stopshere.

# clubing voth the data set with probability 1 and 0 together in a single list in alternate manner, same for probability array.
for el in range (0, max(len(prob0_ds_list), len(prob1_ds_list))):
    if(len(prob0_ds_list) > len(prob1_ds_list)):
        print(" ")
        print("##########################################################################")
        print("Data set 1 > Data set 2.")
        print("Please wait processing data sets.")
        if(el <= len(prob0_ds_list)):
            X.append(prob0_ds_list[el])
            print("X-feature list appended : ", X)
            Y.append(prob_0_y[el])
            print("Y-feature list appended : ", Y)
        else:
            pass
        if(el < len(prob1_ds_list)):
            X.append(prob1_ds_list[el])
            print("X-feature list appended : ", X)
            Y.append(prob_1_y[el])
            print("Y-feature list appended : ", Y)     
        else:
            pass  
    elif(len(prob1_ds_list) > len(prob0_ds_list)):
        print(" ")
        print("##########################################################################")
        print("Data set 2 > Data set 2.")
        print("Please wait processing data sets.")
        print("##########################################################################")
        print(" ")
        if(el < len(prob0_ds_list)):
            X.append(prob0_ds_list[el])
            print("X-feature list appended : ", X)
            Y.append(prob_0_y[el])
            print("Y-feature list appended : ", Y)
        else:
            pass  
        if(el <= len(prob1_ds_list)):
            X.append(prob1_ds_list[el])
            print("X-feature list appended : ", X)
            Y.append(prob_1_y[el])
            print("Y-feature list appended : ", Y)
        else:
            pass
    else:
        print(" ")
        print("##########################################################################")
        print("Data set 1 = Data set 2.")
        print("Please wait processing data sets.")
        print("##########################################################################")
        print(" ")
        X.append(prob0_ds_list[el])
        print("X-feature list appended : ", X)
        Y.append(prob_0_y[el])
        print("Y-feature list appended : ", Y)
        X.append(prob1_ds_list[el])
        print("X-feature list appended : ", X)
        Y.append(prob_1_y[el])
        print("Y-feature list appended : ", Y)
    print("##########################################################################")
    print(" ") 
# clubing ends here.

# saving the combined data set in npy file starts here.
print(" ")
print("##########################################################################")
print("Writing into file.")
np.savez("Solar_Pannel_Crack_Detection.npz", X = X, Y = Y)
print("Writing completed.")
print("##########################################################################")
print(" ")
# saving the combined data set into npy file finished here.