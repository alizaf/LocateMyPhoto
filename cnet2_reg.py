import matplotlib 
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
# from skimage import feature,segmentation
from skimage import color
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import skimage.io
import random
from scipy.ndimage import gaussian_filter
import pandas as pd
from skimage import data,img_as_float
# from skimage.orphology import reconstruction
from os.path import isfile, join
import itertools, shutil, os
from sklearn.decomposition import PCA
from skimage.transform import resize
import pdb
from sklearn.metrics import classification_report,accuracy_score, recall_score
# from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import normalize
import theano

def createarray(filepath):
    p = skimage.io.imread(filepath,as_grey =False)
    # pgray = color.rgb2gray(p)
    X = resize(p,(96,96))
    # pdb.set_trace()
    return X.T.flatten()
# read images
def getfeatures(dirpath, df_folder):
    features_array = np.array([])
    df=pd.DataFrame({'image':[]})
    nn=0
    domain = df_folder.index# [filenames[i] for i in rands]
    nfiles = len(domain)
    for i,f in enumerate(domain):
        if os.path.exists(dirpath+f):
            df.image[i] = createarray(dirpath + f)
        else:
            df_folder.drop(f,inplace=True)
    X = np.vstack(df['image'].values) # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    # features_array = features_array.reshape(nfiles,nfeatures)    
    print('read_images done')
    return X,df_folder#.reset_index(inplace=True)
def readsplit(pathname, csv2read):
    df = pd.DataFrame()
    # for i in csv2read:
        # df = pd.concat([df,pd.read_csv(pathname+i)])
    df = pd.read_csv(pathname+csv2read).dropna()#.reset_index()
    #df = df.iloc[-10000:,:]
    
    testindex = random.sample(df.index, int(df.shape[0])/5)
    testdf = df.loc[testindex]
    df.drop(testindex, inplace = True)

    valindex = random.sample(df.index, int(df.shape[0])/3)
    valdf = pd.DataFrame(df.loc[valindex])
    # pdb.set_trace()
    df.drop(valindex, inplace = True)
    filenames = df['successnames']
    df.set_index('successnames',inplace=True)
    valdf.set_index('successnames',inplace=True)
    testdf.set_index('successnames',inplace=True)

    Xtrain,df = getfeatures(pathname,df)
    ytrain = np.array(df[['lat','lng']]).astype(np.float32)
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)

    Xval, valdf = getfeatures(pathname,valdf )
    yval = np.array(valdf.reset_index()[['lat','lng']]).astype(np.float32)

    Xtest, testdf = getfeatures(pathname,testdf)
    ytest = np.array(testdf.reset_index()[['lat','lng']]).astype(np.float32)
    train_val_test = [Xtrain, Xtest, ytrain, ytest] #Xval, yval, Xtest, ytest]
    return train_val_test
def haversine(self,lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km
def load2d(test=False, cols=None):
    # X, y = load(test=test)
    [X, X_test, y, y_test] = readsplit(pathname,'folderdata.csv')
    # pdb.set_trace()
    X = X.reshape(-1, 3, 96, 96)
    X_test = X_test.reshape(-1, 3, 96, 96)
    y = ((y - SW)*100.).astype(np.float32)
    y_test = ((y_test - SW)*100.).astype(np.float32)
    return X, y, X_test, y_test

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

pathname = './data/photodb41/'

# [X_train, X_test, y_train, y_test] = readsplit(pathname,'folderdata.csv')
# print X_train.shape
# print y_train.shape
# y_train = y_train
# net1.fit(X_train, y_train)
SW = [37.78, -122.41]
# SW = [37.755,  -122.45]

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(5, 5), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=2, output_nonlinearity=None,

    update_learning_rate=theano.shared(np.float32(0.0001), borrow=True),
    update_momentum=theano.shared(np.float32(0.001), borrow=True),

    regression=True,
    on_epoch_finished=[
    AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
    AdjustVariable('update_momentum', start=0.005, stop=0.999),
    ],
    max_epochs=5000,
    verbose=1,
    )

X_train, y_train, X_test, y_test = load2d()  # load 2-d data
# pdb.set_trace()
# skimage.io.imsave('test1.png',(X_train[0].T))
net2.fit(X_train, y_train)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)


y_pred = net2.predict(X_test)

y_rand = np.random.rand(y_test.shape[0],2)
# y_rand = (y_rand*0.025)+[37.775,-122.425]
# y_rand = (y_rand*0.01)+[37.78,-122.41]

dd = np.zeros_like(y_test[:,1])
ddr = np.zeros_like(y_test[:,1])
for i in range(len(y_test)):
    dd[i] = haversine(1, y_pred[i,1], y_pred[i,0],y_test[i,1],y_pred[i,0])
    ddr[i] = haversine(1, y_rand[i,1], y_rand[i,0],y_test[i,1],y_rand[i,0])
print 'random distance=%f versus model distance =%f' %(ddr.mean(),dd.mean())


print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

