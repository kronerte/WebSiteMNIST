from PIL import Image, ImageFilter
import numpy as np
from cnn import *




def MNIST_PREDICTION(image,name):
    im = image
    im = im.convert('L')
    im = im.resize((28,28),Image.ANTIALIAS)
    im = np.asarray(im)
    im = 1-im/255
    im_f = im.flatten()
    predit = predict_cnn(im_f.reshape((1,-1)))
    return image, name + str(predit)



transforms = { 'EMBOSS':lambda im, name: (im.filter(ImageFilter.EMBOSS),name),
               'FIND_EDGES': lambda im, name: (im.filter(ImageFilter.FIND_EDGES), name),
               'MNIST_PREDICTION':MNIST_PREDICTION

}

def do(method, image, name):
    return transforms[method](image, name)
