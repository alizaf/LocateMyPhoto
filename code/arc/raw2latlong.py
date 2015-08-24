import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from collections import Counter
import shutil
from os import listdir
from os.path import isfile, join
from math import radians, cos, sin, asin, sqrt
import numpy as np
import re
import requests
import urllib
from flask import Flask
from flask import request
import time
from random import randint

class getview(object):
    """Pool Layer of a convolutional network """
    def __init__(self, rawdatafile, SW, NE, where2store,dmin, meshsize):
        self.dfraw = pd.read_csv(rawdatafile)
        self.SW = SW
        self.NE = NE
        self.where2store = where2store
        self.dmin = dmin
        self.meshsize = meshsize
    # def raw2latlng(self):
        df_latlng = self.dfraw['Business_Location'].map(lambda x: str(x).split()[-2:])
        df_lat = df_latlng.map(lambda x: x[0][1:-1])
        df_lng = df_latlng.map(lambda x: x[-1][0:-1])
        df_latlng = pd.concat([df_lat,df_lng],1)
        df_latlng.columns = ['lat', 'lng']
        df = df_latlng.convert_objects(convert_numeric=True)
        
        df=df[(df.lat<NE[0]) & (df.lat>SW[0])]
        df=df[(df.lng<NE[1]) & (df.lng>SW[1])]

        self.df_latlng = df

    def info2name(self, lat, lng, angle, label):
        return 'lat%.6f_lng%.6fang%03d_%dlab%d.png' %(lat,lng,angle,label)

    def name2info(self, filename):
        filename = re.findall(r"[^\W\d_]+|\d+.\d+",filename)
        nameinfo = dict()
        for i in range (len(filename)-1):
            nameinfo[filename[i]] = filename[i+1]
        return nameinfo

    def slicepics(self, pathname,  SW, NE, newpath = None):
        onlyfiles = [ f for f in listdir(pathname) if (isfile(join(pathname,f))) & (len(f)>=31) ]
        validfiles = [f for f in onlyfiles if (float(f[3:12])<NE[0]) & (float(f[3:12])>SW[0])
             & (float(f[16:27])<NE[1]) & (float(f[16:27])>SW[1])]
        if newpath:
            if not os.path.exists(newpath):
                os.mkdir(newpath)
    #         moved = [os.rename(pathname+f, newpath+f) for f in validfiles]
            copied = [shutil.copy(pathname+f, newpath+f) for f in validfiles]

        self.pics4slice = validfiles

    #function that gets list of file names and return dataframe with lat lng as columns
    def filenameplot(filenames,plot = 0):
        latlist = [float(filenames[i][3:12]) for i in range(len(filenames))]
        lnglist = [float(filenames[i][16:27]) for i in range(len(filenames))]
        namelist = [filenames[i] for i in range(len(filenames))]


        df = pd.DataFrame(np.transpose([latlist, lnglist,namelist]), columns=['lat','lng','filename'])
        if plot == 1:
            df.plot('lng','lat',kind='scatter',s = 0.3, figsize= [5,5])
        return df
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
    def creatdistinct(self):
        df=self.df_latlng[['lat','lng']]
        
        yy = df['lat'].values
        xx = df['lng'].values
        p = df[['lng', 'lat']].values
        p1 = []
        for i in range(len(yy)-1):
            for j in range (i+1,len(yy)):
                d = haversine(xx[i],yy[i],xx[j],yy[j])
                if d < self.dmin:
                    p1.append(p[i])
                    break
        dfp1 = pd.DataFrame(p1)
        dfp1.columns = df.columns
        df_filtered = pd.concat([df, dfp1])
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered_gpby = df_filtered.groupby(list(df_filtered.columns))
        #get index of unique records
        idx = [x[0] for x in df_filtered_gpby.groups.values() if len(x) == 1]
        #filter
        df_filtered = df_filtered.reindex(idx)
        df_filtered.plot('lng','lat',kind='scatter',figsize=[15,10],s=0.5)

        self.df_filt = df_filtered

        latlinspace = np.linspace(self.SW[0],self.NE[0],self.meshsize[0])
        lnglinspace = np.linspace(self.SW[1],self.NE[1],self.meshsize[1])

        mesh = np.array(np.meshgrid(latlinspace, lnglinspace))

        self.df_filt['label'] = 0
        sortedfiles = []
        for i in range(meshsize[0]):
            for j in range(meshsize[1]):
                inarea = [f for f in onlyfiles if (float(f[3:12])<mesh[:,i+1,j+1[0] & (float(f[3:12])>mesh[:,i,j][0])
             & (float(f[16:27])<mesh[:,i+1,j+1][1]) & (float(f[16:27])>mesh[:,i,j][1])]
                sortedfiles.append(filenameplot(inarea,plot = 0))

        for i in range(len(sortedfiles)-1):
            sortedfiles[i]['label']=i

        dflabeled=sortedfiles[0]

        for i in range(len(sortedfiles)-1):
            dflabeled = pd.concat([dflabeled,sortedfiles[i+1]],axis=0)
        self.df_labeled = dflabeled
        self.df_labeled .to_csv(where2store)


def single_query(self, filename, ii, nsucs, ntries):
    #getfile = urllib.URLopener()
    # print lat,longit
    link='https://maps.googleapis.com/maps/api/streetview?size=300x400&location=%.6f,%.6f&fov=90&heading=%d&pitch=10&key=AIzaSyCsHPCM6nVYgvItOGnmXq17FhvKtjOp44k'%(lat,longit,angle)
    print link
    pathname = self.where2save + infor2name(lat, longit, angle, label)
    urllib.urlretrieve(link, pathname)
    if os.stat(pathname).st_size == 3762 or os.stat(pathname).st_size==7850:
        os.remove(pathname)
        ntries += 1
        return nsucs, ntries
    else:
        try:
            print '%d success!:%d %d Cum Success = %.3f' %(ii, nsucs, ntries, float(nsucs)/ntries)
        except ZeroDivisionError:
            print '%d success! Cum Success = %.3f' %(ii, 0)
        nsucs += 1
        ntries += 1
        return nsucs, ntries

def query(self):
    # df = pd.read_csv('processed.csv')
    # df = df.drop(0,axis=0)
    nsucs = 0
    ntries = 0
    self.df_success = pd.DataFrame()
    angles = [45,135,225,315]#, 45, 120, 135, 210, 225, 300, 315]
    nfiles = os.stat(where2store).st_nlink
    for i in range(df.shape[0]):#nfiles+1,42867)nfiles+10001):#range(df.shape[0]):
        # if isinthebox(df.iloc[i,1],df.iloc[i,2], SW, NE):
        filename = infor2name(self.df_labeled.iloc[i,1],self.df_labeled.iloc[i,2],angles[randint(0,3)], df_labeled['label'][i]
        nsucs_new, ntries = single_query(filename, i , nsucs, ntries)
        if nsucs_new > nsucs:
            nsucs = nsucs_new
            self.df_success.iloc[nsucs] = self.df_labeled[i,:]
            time.sleep(0.0)

