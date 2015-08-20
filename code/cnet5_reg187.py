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
import matplotlib.pyplot as plt


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
    # X = p
    X=p/255.
    if X.shape[0] != picsize:
        X = resize(p,(picsize,picsize))
    return X.T.flatten().astype(np.float32)
# read images
def getfeatures(dirpath, df_folder):
    print 'Trying to load %d files'%df_folder.shape[0]
    features_array = np.array([])
    df=pd.DataFrame({'image':[]})
    nn=0
    domain = df_folder.index# [filenames[i] for i in rands]
    nfiles = len(df_folder.index)
    for i,f in enumerate(df_folder.index):
        if os.path.exists(dirpath+f):
            if i%1000 == 0: print 'Number of files so far:', i
            # print f
            df.image[i] = createarray(dirpath + f)
        else:
            if f in df_folder.index:
                df_folder.drop(f,inplace=True)
    X = np.vstack(df['image'].values) # scale pixel values to [0, 1]
    print('read_images done')
    return X,df_folder#.reset_index(inplace=True)
def readsplit(pathname, csv2read):
    df = pd.DataFrame()
    df = pd.read_csv(pathname+csv2read).dropna()#.reset_index()
    # df_top200 = pd.read_csv('top200.csv').set_index('streetname')
    # df_top200.columns = ['strank'] 
    # df = df.join(df_top200, on='streetname')
    #filter for number of streets
    # df = df[(df.strank <200) & (df.strank > 120)]
    
    testindex = random.sample(df.index, 1000)
    testdf = df.loc[testindex[:1000]]
    df.drop(testindex, inplace = True)

    filenames = df['successnames']
    df.set_index('successnames',inplace=True)
    # valdf.set_index('successnames',inplace=True)
    testdf.set_index('successnames',inplace=True)

    Xtrain,df = getfeatures(pathname,df)
    # Xtrain = transform(Xtrain)
    ytrain = np.array(df[['lat','lng']]).astype(np.float32)
    # Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)

    Xtest, testdf = getfeatures(pathname,testdf)#.iloc[:2000,:])
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
def customize(data,SW_data=None,NE_data=None):
    if SW_data == None:
        SW_data = y_train.min(0)
    if NE_data==None:
        NE_data = y_train.max(0)
    DegDiff = NE_data-SW_data
    data = ((data - SW_data)/DegDiff*100.).astype(np.float32)
    return data
def inverse_customize(data,SW_data,NE_data):
    DegDiff = NE_data-SW_data
    return (data*DegDiff/100.)+SW_data

def scattersave(data1, data2, outpath1, data3=None,data4=None, color1='b',color2='r', s=2):
    plt.scatter(data1,data2, s=s, color=color1)
    if data3!= None:
        plt.scatter(data3, data4,color=color2 )
    plt.savefig(outpath1)

def histogramsave(data, outpath1, color='b', bins=10):
    fig = plt.figure()
    try:
        plt.hist(data, bins=bins, color=color)
        plt.ylim(0, 500)
        plt.xlim(0, 10)
        plt.savefig(outpath1)
    except:
        pass
    plt.close(fig)

def load2d(test=False, cols=None):
    # X, y = load(test=test)
    [X, X_test, y, y_test] = readsplit(pathname,csv2read)
    X = X.reshape(-1, 3, picsize, picsize)
    X_test = X_test.reshape(-1, 3, picsize, picsize)
    y = y.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X, y, X_test, y_test

# def transform(Xb):
#     # Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
#     # Flip half of the images in this batch at random:
#     bs = Xb.shape[0]
#     indices = np.random.choice(bs, bs / 2, replace=False)
#     Xb[indices] = Xb[:, :, ::-1]
#     return Xb


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch'] 
        if self.ls is None:
            if epoch <1000:
                self.ls = np.linspace(self.start, self.stop, nn.max_epochs/2)
            else:
                self.ls = 1e-7
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class HaverSineDist(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        if epoch%2 == 0:
            f = getattr(nn, self.name)
            y_pred = inverse_customize(f(X_test), SW,NE)
            d = np.zeros_like(y_test[:,1])
            for i in range(len(y_test)):
                d[i] = haversine(y_pred[i,1], y_pred[i,0],y_test[i,1],y_test[i,0])
            dft = pd.DataFrame({epoch : d, 'lat':y_pred[:,0], 'lng':y_pred[:,1]})
            dft.to_csv(outpath+'latlng%d.csv'%epoch)
            # pdb.set_trace()
            dd.append(dft)

            fig = plt.figure()
            plt.scatter(dft[dft[epoch]<1.].lng, dft[dft[epoch]<1.].lat, s=10,color='b')
            plt.scatter(dft[dft[epoch]>1.].lng, dft[dft[epoch]>1.].lat, s=3,color='r')
            # scattersave(y_test[:,0],y_test[:,1], outpath+'latlng_plot%d.png'% epoch,data3=y_pred[:,0],data4=y_pred[:,1])
            plt.scatter(y_test[:,1],y_test[:,0], s=2)
            plt.savefig(outpath+'latlng_plot%05d.png'% epoch)
            plt.close(fig)

            histogramsave(d, outpath+'distance_hist%05d.png'% epoch)

            train_loss = np.array([i["train_loss"] for i in net3.train_history_])/100.
            valid_loss = np.array([i["valid_loss"] for i in net3.train_history_])/100.
            fig = plt.figure()
            plt.plot(train_loss, linewidth=3, label="train")
            plt.plot(valid_loss, linewidth=3, label="valid")
            plt.grid()
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.ylim(1e-1, 1e1)
            plt.yscale("log")
            plt.savefig(outpath+'train_val_loss.png')
            plt.close(fig) 
####################################################################################################################
picsize = 96

net3 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, picsize, picsize),
    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    dropout1_p=0.0002,  # !
    conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    dropout2_p=0.0004,  # !
    conv3_num_filters=64, conv3_filter_size=(5, 5), pool3_pool_size=(2, 2),
    dropout3_p=0.0006,  # !
    hidden4_num_units=500, 
    dropout4_p=0.001,  # !
    hidden5_num_units=500,
    output_num_units=2, output_nonlinearity=None,
    # update = adagrad,
    update_learning_rate=theano.shared(np.float32(0.00001), borrow=True),
    update_momentum=theano.shared(np.float32(0.01), borrow=True),
    objective_loss_function=None,
    regression=True,
    on_epoch_finished=[
    AdjustVariable('update_learning_rate', start=5e-6, stop=1e-7),
    AdjustVariable('update_momentum', start=0.0001, stop=0.002),
    HaverSineDist('predict'),
    ],
    max_epochs=2000,
    verbose=1,
    )

####################################################################################################################
import socket
ip = socket.getfqdn()
outpath = './model_outputs%s_%s_60/'%(picsize,ip)
# shutil.copy('cnet5_reg133.py',outpath[2:])

if not os.path.exists(outpath):
    os.makedirs(outpath)

pathname = './data/photodb_MainST600_40m03r/'#photodb_MainST600_40m03r/'#photodb_MainST100_25m1r#photodb_MainST_NE
csv2read = 'folderdata.csv'


# #box northeast
SW_northeast = np.array([37.76, -122.43])
NE_northeast = np.array([37.815, -122.38])
#box including everithing
SW_sf = np.array([37.707875, -122.518624])
NE_sf = np.array([37.815086, -122.378205])
#box ~1km:
SW_1k = np.array([37.77, -122.42])
NE_1k = np.array([37.79, -122.40])

SW = SW_sf
NE = NE_sf
####################################################################################################################

csv2save = 'folderdata_SW%d_%dNE%d_%d.csv' %(SW[0],SW[1],NE[0],NE[1])
csv2read,SW_rdata, NE_rdata = inabox_select(SW,NE,pathname,csv2read,csv2save)


X_train, y_train, X_test, y_test = load2d()  # load 2-d data

y_train_trfm = customize(y_train,SW,NE)
y_test_trfm = customize(y_test,SW,NE)
df_ytest = pd.DataFrame(y_test)
df_ytest.to_csv(outpath+'y_test_df')
dd=[]
# print 'size of the lat-long box for the raw data (SW%f-%f,NE%f,%f): L%f x W%f' %\
#     (SW_rdata[0],SW_rdata[1],NE_rdata[0],NE_rdata[1], haversine(SW_rdata[1],SW_rdata[0],NE_rdata[1],SW_rdata[0]),\
#         haversine(SW_rdata[1],SW_rdata[0],SW_rdata[1],NE_rdata[0]))
# print 'size of the lat-long box for the selected data (SW%f-%f,NE%f,%f): L%f x W%f' %\
#     (SW_data[0],SW_data[1],NE_data[0],NE_data[1], haversine(SW_data[1],SW_data[0],NE_data[1],SW_data[0]),\
#         haversine(SW_data[1],SW_data[0],SW_data[1],NE_data[0]))

print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
    X_train.shape, X_train.min(), X_train.max()))

net3.fit(X_train, y_train_trfm)

with open('net64_1_SF_%d_25_1r_3.pickle'%picsize,'wb') as f:#%(SW[0],SW[1],NE[0],NE[1]), 'wb') as f:
    pickle.dump(net3, f, -1)

y_pred_trfm = net3.predict(X_test)
y_pred = inverse_customize(y_pred_trfm,SW,NE)

print 'model mean distance =%f, with %d points within 1 km (test size=%d'%(dd[-1].mean(),(dd[-1]<1).sum(),y_test.shape[0])
print 'size of the lat-long box for the raw data (SW%f-%f,NE%f,%f): L%f x W%f' %\
    (SW_rdata[0],SW_rdata[1],NE_rdata[0],NE_rdata[1], haversine(SW_rdata[1],SW_rdata[0],NE_rdata[1],SW_rdata[0]),\
        haversine(SW_rdata[1],SW_rdata[0],SW_rdata[1],NE_rdata[0]))
df_dd = pd.DataFrame({'dist_epochs':dd})
df_dd.to_csv(outpath + 'distance_epochs.csv')


# print 'size of the lat-long box for the selected data (SW%f-%f,NE%f,%f): L%f x W%f' %\
#     (SW_data[0],SW_data[1],NE_data[0],NE_data[1], haversine(SW_data[1],SW_data[0],NE_data[1],SW_data[0]),\
#         haversine(SW_data[1],SW_data[0],SW_data[1],NE_data[0]))


# plt.scatter(y_train[:,0],y_train[:,1])
# plt.savefig(outpath+'test16.png')
# plt.close(fig)

# plt.figure()
# plt.hist(dd)
# plt.savefig(outpath+'test15.png')


