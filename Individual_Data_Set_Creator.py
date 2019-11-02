import numpy as np
import matplotlib.image as mpimg
import glob

push_list = []

def converting_image_into_1d_nparray(this_image, row, column):
    """
        This function will take three values and convert the image into 1d numpy array.
        1) this_image = read_image ie image that is been selected for conversion at present.
        2) row  = r ie no of rows in selected image.
        3) column = c ie no of columns in selected image.
    """
    image_in_list = []
    for i in range(0, row):
        for j in range(0, column):
            image_in_list.append(this_image[i, j])       
    ld_nparray_of_image = np.array(image_in_list)
    image_in_list.clear()
    return ld_nparray_of_image


def converting_image_into_3d_nparray(this_image, row, column):
    """
        This function will take three values and convert the image into numpy array.
        1) this_image = read_image ie image that is been selected for conversion at present.
        2) row  = r ie no of rows in selected image.
        3) column = c ie no of columns in selected image.
    """
    image_in_list = []
    
    for i in range(0, row):
        for j in range(0, column):
            image_in_list.append(this_image[i, j])
            
    nparray_of_image = np.array(image_in_list).reshape(300, 300)
    image_in_list.clear()
    return nparray_of_image


while True:
    print(" ")
    print("##########################################################################")
    print(" ")
    print("Enter 1 for converting generating 1D np-array.")
    print("Enter 2 for converting generating 3D np-array.")
    option = input("Your Choice : ")
    if(option == "1"):
        path = '*.png'
        files = glob.glob(path)
        for name in files:
            print("##########################################################################")
            print("*******************************************************************************")
            print("reading image : ", name)
            read_image = mpimg.imread(name)
            r, c = read_image.shape
            print("Size for read image %s is : %dx%d" % (name, r, c))    
            np_array = converting_image_into_1d_nparray(this_image = read_image, row = r, column = c)
            print("Size of numpy array for image %s is : %d" % (name, np_array.size))
            print("Updating push list with new 1d np array of image")
            push_list.append(np_array)
            print("Push list updated with new 1d np array of image")
            print("Total images in the push list to create csv dump : ", len(push_list))
            print("*******************************************************************************")
            # saving individual data set in npy file starts here.
            print(" ")
            print("##########################################################################")
            print("Writing into file.")
            np.savez("Image_with_prpbability_1D_np-array.npy", push_list)
            print("Writing completed.")
            print("##########################################################################")
            # saving the individual data set into npy file finished here.
        print("##########################################################################")
        break
    elif(option == "2"):
        path = '*.png'
        files = glob.glob(path)
        for name in files:
            print("##########################################################################")
            print("*******************************************************************************")
            print("reading image : ", name)
            read_image = mpimg.imread(name)
            r, c = read_image.shape
            print("Size for read image %s is : %dx%d" % (name, r, c))    
            np_array = converting_image_into_3d_nparray(this_image = read_image, row = r, column = c)
            print("Size of numpy array for image %s is : %d" % (name, np_array.size))
            print("Updating push list with new 3d np array of image")
            push_list.append(np_array)
            print("Push list updated with new 3d np array of image")
            print("Total images in the push list to create csv dump : ", len(push_list))
            print("*******************************************************************************")
            # saving individual data set in npy file starts here.
            print(" ")
            print("##########################################################################")
            print("Writing into file.")
            np.savez("Image_with_prpbability_3D_np-array.npz", push_list)
            print("Writing completed.")
            print("##########################################################################")
            # saving the individual data set into npy file finished here.
            print("##########################################################################")
            break
    else:
        print("Select from listed options.")
        continue






