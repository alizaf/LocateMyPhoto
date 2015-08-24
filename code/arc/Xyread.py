from skimage import feature
from skimage import color
from skimage import segmentation

from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import skimage
import skimage.io
import matplotlib.pylab as plt
import random
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from os.path import isfile, join
import itertools, shutil, os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def createarray(filepath):
    p = skimage.io.imread(filepath)
    pgray = color.rgb2gray(p)
    X = pgray
    X = feature.canny(pgray,sigma=2)


    return X.flatten()

# read images
def getfeatures(dirpath, filenames):
	features_array = []
	nn=0
	# rands = np.random.random_integers(0,1500,100)
	domain = filenames # [filenames[i] for i in rands]
	print domain
	for i,f in enumerate(domain):
		temp = createarray(dirpath +f)
		print temp
		features_array.append(temp)
		#print len(features_array)
	features_array = np.array(features_array)    
	print('read_images done')
	return features_array

def readsplit(pathname, csv2read):
    df = pd.read_csv(pathname+csv2read)
    testindex = random.sample(df.index, int(df.shape[0])/5)
    testdf = df.loc[testindex]
    df.drop(testindex, inplace = True)

    valindex = random.sample(df.index, int(df.shape[0])/3)
    valdf = df.loc[valindex]
    df.drop(valindex, inplace = True)
    filenames = df['successnames']

    Xtrain = getfeatures(pathname,df.successnames )
    ytrain = np.array(df.reset_index()['label']).astype(np.int32)

    Xval = getfeatures(pathname,valdf.successnames )
    yval = np.array(valdf.reset_index()['label']).astype(np.int32)

    Xtest = getfeatures(pathname,testdf.successnames )
    ytest = np.array(testdf.reset_index()['label']).astype(np.int32)
    train_val_test = [Xtrain, Xtest, ytrain, ytest]#Xval, yval, Xtest, ytest]
    return train_val_test
pathname = './photodb11/'
readsplit(pathname,'imagedata.csv')
	# features_array = features_array*0.99
	# pca = PCA(n_components=5000)
	# feature_pca = pca.fit_transform(features_array)
