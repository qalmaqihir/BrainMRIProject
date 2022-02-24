import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Transforms
img = Image.open('GrayscaleImages/12.gif')

def log_transform(img):
    # img.show()
    # img is converted into an ndarray
    b = np.asarray(img)
    # #convert b to type float
    b1 = b.astype(float)
    # Maximum value in b1 is determine
    b3 = np.max(b1)
    # performing the log transformation
    c = (255.0 * np.log(1 + b1)) / np.log(1 + b3)
    # c is converted to type int
    transformed = c.astype(np.uint8)
    return transformed

    # c1 is converted from ndaraay to image
    # d = Image.fromarray(c1)
    # d.show()
    # d.save('logtransfrom.gif')

    # img.show()
    # # img is converted into ndarray
    # b = scipy.misc(img)
    # # ndarray is converted to float type
    # b1=b.astype(float)
    #
    #
    # # Max value in b1 is found
    # b2= np.max(b1)
    #
    # # performing the log transformation
    # c = (255.0 * np.log(1+b1))/np.log(1+b2)
    #
    # # c is converted from float ndarray to int
    # c1= c.asarray(int)
    #
    # # Converting back to image
    # im=Image.fromarray(c1)
    # im.save('test_log_output.png')



# log_transform(img)


def contrast_stretching(img):
    im=np.asarray(img)

    # finding the min and max
    b = im.max()
    a=im.min()

    print(f"Min pixel Value {a}\t Max pixel Value {b}")

    # converting im to float
    im1=im.astype(float)

    #Contrast stretching transformation
    transformed = 255*(im1-a)/(b-a)
    return transformed


    # converting c back to ndarray image
    # im2=Image.fromarray(c)
    # im2.save("Contrast_stretching_output.gif")
    # im2.show()




# contrast_stretching(img)


# For all the Grayscale images
i=0
for filename in glob.glob('/home/jawad/Downloads/FYP/Codes/Code and Data/GrayscaleImages/*.gif'):
    i+=1
    img = Image.open(filename)
    # Applying the transforms
    l=log_transform(img)
    # converting the output array to image
    l=Image.fromarray(l)
    c =contrast_stretching(l)

    # Saving the images
    name='Transformed_Images/%d.gif'%i
    # cv2.imwrite(name,Image.fromarray(m))
    p=Image.fromarray(c)
    p.save(name)

