import matplotlib
matplotlib.use('Agg')
from sklearn.cross_validation import train_test_split
from skimage import color
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import skimage.io
import random
from scipy.ndimage import gaussian_filter
import pandas as pd
from skimage import data, img_as_float
from os.path import isfile, join
import itertools
import shutil
import os
from sklearn.decomposition import PCA
from skimage.transform import resize
import pdb
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.utils import shuffle


def inabox_select(SW, NE, pathname, csv2read, csv2save):
    df = pd.read_csv(pathname + csv2read).dropna()  # .reset_index()
    mask = (df.lat > SW[0]) & (df.lat < NE[0]) & (
        df.lng > SW[1]) & (df.lng < NE[1])
    df_inbox = df[mask]
    df_inbox.to_csv(pathname + csv2save)
    SW_data = np.array([df.lat.min(), df.lng.min()])
    NE_data = np.array([df.lat.max(), df.lng.max()])
    return csv2save, SW_data, NE_data


def createarray(filepath):
    try:
        p = skimage.io.imread(filepath, as_grey=False)
        X = p / 255.
        X = resize(p, (picsize[0], picsize[0]))
        return X.T.flatten().astype(np.float32)
    except:
        print 'not found:', filepath
        return


def getfeatures(dirpath, df_folder):
    print 'Trying to load %d files' % df_folder.shape[0]
    features_array = np.array([])
    df = pd.DataFrame({'image': []})
    nn = 0
    domain = df_folder.index
    nfiles = len(df_folder.index)
    for i, f in enumerate(df_folder.index):
        if os.path.exists(dirpath + f):
            if i % 1000 == 0:
                print 'Number of files so far:', i
            # print f
            df.image[i] = createarray(dirpath + f)
        else:
            if f in df_folder.index:
                df_folder.drop(f, inplace=True)
    X = np.vstack(df['image'].values)  # scale pixel values to [0, 1]
    print('read_images done')
    return X, df_folder


def readsplit(pathname, csv2read):
    df = pd.DataFrame()
    df = pd.read_csv(pathname + csv2read).dropna()
    df_top200 = pd.read_csv('top200.csv').set_index('streetname')
    df_top200.columns = ['strank']
    df = df.join(df_top200, on='streetname')

    testindex = random.sample(df.index, 800)
    testdf = df.loc[testindex[:800]]
    df.drop(testindex, inplace=True)

    filenames = df['successnames']
    df.set_index('successnames', inplace=True)
    testdf.set_index('successnames', inplace=True)

    Xtrain, df = getfeatures(pathname, df)
    ytrain = np.array(df[['lat', 'lng']]).astype(np.float32)

    Xtest, testdf = getfeatures(pathname, testdf)
    ytest = np.array(testdf.reset_index()[['lat', 'lng']]).astype(np.float32)
    train_val_test = [Xtrain, Xtest, ytrain, ytest]
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
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def customize(data, SW_data=None, NE_data=None):
    if SW_data == None:
        SW_data = y_train.min(0)
    if NE_data == None:
        NE_data = y_train.max(0)
    DegDiff = NE_data - SW_data
    data = ((data - SW_data) / DegDiff * 100.).astype(np.float32)
    return data


def inverse_customize(data, SW_data, NE_data):
    DegDiff = NE_data - SW_data
    return (data * DegDiff / 100.) + SW_data


def scattersave(data1, data2, outpath1, data3=None, data4=None, color1='b', color2='r', s=2):
    plt.scatter(data1, data2, s=s, color=color1)
    if data3 != None:
        plt.scatter(data3, data4, color=color2)
    plt.savefig(outpath1)


def histogramsave(data, outpath1, color='b', bins=10):
    fig = plt.figure()
    try:
        plt.hist(data, bins=bins, color=color)
        plt.ylim(0, 600)
        plt.xlim(0, 10)
        plt.savefig(outpath1)
    except:
        pass
    plt.close(fig)


def load2d(pathname, csv2read, test=False, cols=None):
    [X, X_test, y, y_test] = readsplit(pathname, csv2read)
    X = X.reshape(-1, 3, picsize[0], picsize[1])
    X_test = X_test.reshape(-1, 3, picsize[0], picsize[1])
    y = y.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X, y, X_test, y_test


def transform(Xb):
    # Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
    # Flip half of the images in this batch at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs / 2, replace=False)
    Xb[indices] = Xb[:, :, ::-1]
    return Xb
