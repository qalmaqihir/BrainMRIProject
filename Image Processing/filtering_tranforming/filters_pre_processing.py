# Imports
import glob

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.misc
# Filters
# Medain Filter
def medain_filters(img):
    filtered=scipy.ndimage.median_filter(img,size=5, footprint=None, mode='reflect',output=None, cval=0.0,origin=0)
    return filtered

def laplacian_filters(img):
    filtered = scipy.ndimage.laplace(img,)
    return filtered

def prewitt_filters(img):
    filtered = scipy.ndimage.prewitt(img)
    return filtered


# img = Image.open('GrayscaleImages/11.gif')
# m = medain_filters(img)
# print(type(m))
# im = Image.fromarray(m)
# im.show()
# cv2.imwrite('output_m.png',im)



# For all the Grayscale images
i=0
for filename in glob.glob('/home/jawad/Downloads/FYP/Codes/Code and Data/GrayscaleImages/*.gif'):
    i+=1
    img = Image.open(filename)
    # Applying the three filters
    m=medain_filters(img)
    # converting the numpy array to image
    m=Image.fromarray(m)
    l=laplacian_filters(m)
    # # converting the numpy array to image
    l=Image.fromarray(l)
    p=prewitt_filters(l)

    # Saving the images
    name='Filtered_Images/%d.gif'%i
    # cv2.imwrite(name,Image.fromarray(m))
    p=Image.fromarray(p)
    p.save(name)





