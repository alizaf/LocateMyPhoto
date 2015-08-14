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
from lasagne.updates import nesterov_momentum, sgd, momentum, adagrad
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import normalize, StandardScaler
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import theano
import cPickle as pickle


def inabox_select(SW,NE,pathname,csv2read,csv2save):
    df = pd.read_csv(pathname+csv2read).dropna()#.reset_index()
    mask = (df.lat>SW[0]) & (df.lat<NE[0]) & (df.lng>SW[1]) & (df.lng<NE[1])
    df_inbox = df[mask]
    df_inbox.to_csv(pathname + csv2save)
    SW_data = np.array([df.lat.min(),df.lng.min()])    
    NE_data = np.array([df.lat.max(),df.lng.max()])

    return csv2save,SW_data, NE_data

def createarray(filepath):
    try:
        p = skimage.io.imread(filepath,as_grey =False)
    except:
        print filepath

    X=p
    # if X.shape[0] != 160:
    #     # pgray = color.rgb2gray(p)
    #     X = resize(p,(160,160))
    # pdb.set_trace()
    return X.T.flatten()
# read images
def getfeatures(dirpath, df_folder):
    features_array = np.array([])
    df=pd.DataFrame({'image':[]})
    nn=0
    domain = df_folder.index# [filenames[i] for i in rands]
    nfiles = len(df_folder.index)
    for i,f in enumerate(df_folder.index):
        if os.path.exists(dirpath+f):
            # print f
            df.image[i] = createarray(dirpath + f)
        else:
            # pdb.set_trace()
            if f in df_folder.index:
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
    # df = df.iloc[-10:,:]
    
    testindex = random.sample(df.index, int(df.shape[0]/10))
    testdf = df.loc[testindex]
    df.drop(testindex, inplace = True)

    # valindex = random.sample(df.index, int(df.shape[0])/3)
    # valdf = pd.DataFrame(df.loc[valindex])
    # pdb.set_trace()
    # df.drop(valindex, inplace = True)
    filenames = df['successnames']
    df.set_index('successnames',inplace=True)
    # valdf.set_index('successnames',inplace=True)
    testdf.set_index('successnames',inplace=True)

    Xtrain,df = getfeatures(pathname,df)
    ytrain = np.array(df[['lat','lng']]).astype(np.float32)
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    # Xtrain = Xtrain[:2000,:]
    # ytrain = ytrain[:2000,:]

    # Xval, valdf = getfeatures(pathname,valdf )
    # yval = np.array(valdf.reset_index()[['lat','lng']]).astype(np.float32)

    Xtest, testdf = getfeatures(pathname,testdf.iloc[:500,:])
    ytest = np.array(testdf.reset_index()[['lat','lng']]).astype(np.float32)
    train_val_test = [Xtrain, Xtest, ytrain, ytest] #Xval, yval, Xtest, ytest]
    return train_val_test

def haversine(lon1, lat1, lon2, lat2):
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
    [X, X_test, y, y_test] = readsplit(pathname,csv2read)
    # pdb.set_trace()
    X = X.reshape(-1, 3, 160, 160)
    X_test = X_test.reshape(-1, 3, 160, 160)
    y = y.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # y = y.astype(np.float32)
    # y_test = y_test.astype(np.float32)
    # y = ((y - SW)*100./degdiff).astype(np.float32)
    # y_test = ((y_test - SW)*100./degdiff).astype(np.float32)
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
    input_shape=(None, 3, 160, 160),
    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(5, 5), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=2, output_nonlinearity=None,
    # update = adagrad,
    update_learning_rate=theano.shared(np.float32(0.0000005), borrow=True),
    update_momentum=theano.shared(np.float32(0.0005), borrow=True),
    objective_loss_function=None,
    regression=True,
    # on_epoch_finished=[
    # AdjustVariable('update_learning_rate', start=0.001, stop=0.00001),
    # AdjustVariable('update_momentum', start=0.01, stop=0.099),
    # ],
    max_epochs=500,
    verbose=1,
    )


pathname = './data/photodb_MainST200_25m/'
csv2read = 'folderdata.csv'

# #box northeast
SW_northeast = np.array([37.755,  -122.45])
NE_northeast = np.array([37.81, -122.38])
#box including everithing
SW_sf = np.array([37.7, -122.5])
NE_sf = np.array([37.9, -122.3])
#box ~1km:
SW_1k = np.array([37.785, -122.405])
NE_1k = np.array([37.79, -122.40])


SW = SW_sf
NE = NE_sf
degdiff = NE-SW

csv2save = 'folderdata_SW%d_%dNE%d_%d.csv' %(SW[0],SW[1],NE[0],NE[1])
csv2read,SW_data, NE_data = inabox_select(SW,NE,pathname,csv2read,csv2save)

X_train, y_train, X_test, y_test = load2d()  # load 2-d data

print 'size of the lat-long box for the raw data (SW%f-%f,NE%f,%f): L%f x W%f' %\
    (SW_data[0],SW_data[1],NE_data[0],NE_data[1], haversine(SW_data[1],SW_data[0],NE_data[1],SW_data[0]),\
        haversine(SW_data[1],SW_data[0],SW_data[1],NE_data[0]))
print 'size of the lat-long box for the selected data (SW%f-%f,NE%f,%f): L%f x W%f' %\
    (SW[0],SW[1],NE[0],NE[1], haversine(SW[1],SW[0],NE[1],SW[0]),\
        haversine(SW[1],SW[0],SW[1],NE[0]))

print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))
print("y_train.shape == {}; y_test.shape == {:.3f}".format(
    y_train.shape, y_test.shape))

ss = StandardScaler()
y_train_transformed = ss.fit_transform(y_train)
y_test_transformed = ss.fit_transform(y_test)
# pdb.set_trace()
# skimage.io.imsave('test2.png',(X_train[0]))
net2.fit(X_train, y_train_transformed)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
with open('net4_1_SW%d_%dNE%d_%d.pickle'%(SW[0],SW[1],NE[0],NE[1]), 'wb') as f:
    pickle.dump(net2, f, -1)

y_pred_transformed = net2.predict(X_test)
y_rand = ((np.random.rand(y_test.shape[0],2)-1)*2+ss.mean_)*ss.std_

y_train_inv = ss.inverse_transform(y_train_transformed)
y_test_inv = ss.inverse_transform(y_test_transformed)
y_pred = ss.inverse_transform(y_pred_transformed)

dd = np.zeros_like(y_test[:,1])
ddr = np.zeros_like(y_test[:,1])

import matplotlib.pyplot as plt

for i in range(len(y_test)):
    dd[i] = haversine(y_pred[i,1], y_pred[i,0],y_test[i,1],y_test[i,0])
    # ddr[i] = haversine(1, y_rand[i,1], y_rand[i,0],y_test[i,1],y_rand[i,0])
print 'model mean distance =%f, with %d points within 1 km (test size=%d'%(dd.mean(),(dd<1).sum(),y_test.shape[0])
print 'size of the lat-long box for the raw data (SW%f-%f,NE%f,%f): L%f x W%f' %\
    (SW_data[0],SW_data[1],NE_data[0],NE_data[1], haversine(SW_data[1],SW_data[0],NE_data[1],SW_data[0]),\
        haversine(SW_data[1],SW_data[0],SW_data[1],NE_data[0]))
print 'size of the lat-long box for the selected data (SW%f-%f,NE%f,%f): L%f x W%f' %\
    (SW[0],SW[1],NE[0],NE[1], haversine(SW[1],SW[0],NE[1],SW[0]),\
        haversine(SW[1],SW[0],SW[1],NE[0]))

plt.hist(dd)
plt.savefig('test13.png')



