import matplotlib 
matplotlib.use('Agg')
import pandas as pd
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
import time
import random
from random import randint
import pdb
from sklearn.neighbors import NearestNeighbors

random.seed(100)

# getview collects images from google streetview API, using preprocessed lat long point that are stored in uniform_latlng_xxxx
class getview(object):
    """Pool Layer of a convolutional network """
    def __init__(self, rawdatafile, SW, NE, where2store,ready2serve = False):
        self.ready2serve = ready2serve
        self.dfraw = pd.read_csv(rawdatafile)
        self.SW = SW
        self.NE = NE
        self.where2store = where2store
        if not os.path.exists(self.where2store):
            os.mkdir(self.where2store)
        else:
            print 'Directory exists!'
            ans = raw_input('continue?! [y/n]')
            if ans == 'n' : raise
        if ready2serve:
            self.df_latlng = self.dfraw
        else:
            self.simpleclean()
            splitbystreet()
    #Preprocessing of lat-long data
    def simpleclean():
        df_latlng = self.dfraw['Business_Location'].map(lambda x: str(x).split()[-2:])
        df_lat = df_latlng.map(lambda x: x[0][1:-1])
        df_lng = df_latlng.map(lambda x: x[-1][0:-1])
        df_BL =self.df_raw['Business_Location']
        df_latlng = pd.concat([df_lat,df_lng],1)
        df_latlng.columns = ['BL','lat', 'lng']
        df = df_latlng.convert_objects(convert_numeric=True)
        df['BL'] = df_BL
        df=df[(df.lat<NE[0]) & (df.lat>SW[0])]
        df=df[(df.lng<NE[1]) & (df.lng>SW[1])]
        self.df_latlng = df
    #Create name of image using specified info
    def info2name(self, lat, lng, angle):
        return 'lat%.6f_lng%.6fang%s.png' %(lat,lng,angle)
    # Given name, returns image info
    def name2info(self, filename):
        filename = re.findall(r"[^\W\d_]+|\d+.\d+",filename)
        nameinfo = dict()
        for i in range (len(filename)-1):
            nameinfo[filename[i]] = filename[i+1]
        return nameinfo
    # Make model limited to smaller area defined by SWslice and NEslice
    def slicepics(self, pathname,  SWslice, NEslice, newpath = None):
        onlyfiles = [ f for f in listdir(pathname) if (isfile(join(pathname,f))) & (len(f)>=31) ]
        validfiles = [f for f in onlyfiles if (float(f[3:12])<NEslice[0]) & (float(f[3:12])>SWslice[0])
             & (float(f[16:27])<NEslice[1]) & (float(f[16:27])>SWslice[1])]
        if newpath:
            if not os.path.exists(newpath):
                os.mkdir(newpath)
            copied = [shutil.copy(pathname+f, newpath+f) for f in validfiles]

        self.pics4slice = validfiles

    # Gets list of file names and return dataframe with lat lng as columns
    def filenameplot(filenames,plot = 0):
        latlist = [float(filenames[i][3:12]) for i in range(len(filenames))]
        lnglist = [float(filenames[i][16:27]) for i in range(len(filenames))]
        namelist = [filenames[i] for i in range(len(filenames))]


        df = pd.DataFrame(np.transpose([latlist, lnglist,namelist]), columns=['lat','lng','filename'])
        if plot == 1:
            df.plot('lng','lat',kind='scatter',s = 0.3, figsize= [5,5])
        return df
    # converting lat long distance to actual distance (km)
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
    # Regularizing request points 
    def creatdistinct(self,dmin,validate=False):
        self.validate = validate
        self.dmin = dmin
        if self.ready2serve:
            return
        df=self.df_latlng[['lat','lng']]
        if validate:
            df = pd.read_csv(self.where2store+'validated.csv').dropna().reset_index()
            df.set_index('successnames',inplace=True)
            domain = df.index
            for i,f in enumerate(domain):
                if not os.path.exists(self.where2store+f):
                    df.drop(f,inplace=True)
            df.reset_index(inplace=True)
            self.df_filt = df
            self.df_alter = df

        else:

            yy = df['lat'].values
            xx = df['lng'].values
            p = df[['lat', 'lng']].values
            p1 = []
            dx = self.haversine(self.SW[1],self.SW[0],self.NE[1],self.SW[0])
            dy = self.haversine(self.SW[1],self.SW[0],self.SW[1],self.NE[0])
            meshsiz = [int(dx/dmin),int(dy/dmin)]
            latlinspace = np.linspace(self.SW[0],self.NE[0],meshsiz[0]+1)
            lnglinspace = np.linspace(self.SW[1],self.NE[1],meshsiz[1]+1)
            self.df_filt = pd.DataFrame(np.empty([1,len(df.columns)]), columns=[df.columns])
            self.df_alter = pd.DataFrame(np.empty([1,len(df.columns)]), columns=[df.columns])
            self.df_filt.drop(0,inplace=True)
            self.df_alter.drop(0,inplace=True)
            mesh = np.array(np.meshgrid(latlinspace, lnglinspace))
            for i in range(meshsiz[1]):
                if i>0 :print i*meshsiz[0]+j,'out of', meshsiz[0]*meshsiz[1]
                for j in range(meshsiz[0]):
                    y_ne = mesh[:,i+1,j+1][0]
                    y_sw = mesh[:,i,j][0]
                    x_ne = mesh[:,i+1,j+1][1]
                    x_sw = mesh[:,i,j][1]
                    self.df_alter.loc[i*(meshsiz[0]-1)+j] = [np.mean([y_ne,y_sw]), np.mean([x_ne,x_sw])]
                    mask = (df.lat<mesh[:,i+1,j+1][0]) & (df.lat>mesh[:,i,j][0]) \
                    & (df.lng<mesh[:,i+1,j+1][1]) & (df.lng>mesh[:,i,j][1])
                    ixlist = df[mask]
                    if ixlist.shape[0] > 0:
                        self.df_filt.loc[i*(meshsiz[0]-1)+j] = df.loc[ixlist.index[0]]
    # single queries to API
    def single_query(self, lat, lng, angle, label, ii):
        link='https://maps.googleapis.com/maps/api/streetview?size=%dx%d&location=%.6f,%.6f&\
        fov=%d&heading=%d&pitch=%d&source=outdoor&key=AIzaSyCFs1WFdFRhxsxqFMMKLrg2q1xcuaIFc40'%(self.picsize[0],\
         self.picsize[1], lat,lng,self.fov,angle,self.pitch)
        print self.info2name(lat, lng, self.angles)

        pathname = self.where2store + self.filename
        urllib.urlretrieve(link, pathname)
        if os.stat(pathname).st_size in self.errsize:
            os.remove(pathname)
            print 'attempt:', self.attempts
        else:
            try:
                print '%d success!:%d %d Cum Success = %.3f'%(ii,self.nsucs,self.ntries,float(self.nsucs)/self.ntries)
            except ZeroDivisionError:
                print '%d success! Cum Success = %.3f' %(ii, 0)
            self.success = True

    #location based labeling of images for classification
    def relabel(self,dfl):
        latlinspace = np.linspace(self.SW[0],self.NE[0],self.meshsize[0]+1)
        lnglinspace = np.linspace(self.SW[1],self.NE[1],self.meshsize[1]+1)
        mesh = np.array(np.meshgrid(latlinspace, lnglinspace))
        self.df_goal['label'] = 0
        for i in range(self.meshsize[0]):
            for j in range(self.meshsize[1]):

                mask = (self.df_goal.lat<mesh[:,i+1,j+1][0]) & (self.df_goal.lat>mesh[:,i,j][0]) \
                & (self.df_goal.lng<mesh[:,i+1,j+1][1]) & (self.df_goal.lng>mesh[:,i,j][1])
                ixlist = self.df_goal[mask]
                self.current_label += 1
                for ii in ixlist.index:
                    self.df_goal.loc[ii,'label'] = self.current_label
    # Creates requests and collects images
    def query(self,meshsize, picsize,angles,errsize,nattempts,fov,pitch,full = False):
        self.errsize = errsize
        self.meshsize = meshsize
        self.picsize = picsize
        self.current_label = 0
        self.pitch = pitch
        self.fov = fov

        if self.ready2serve:
            self.df_goal = self.df_latlng
        elif full:
            self.df_goal = self.df_alter
        else:
            self.df_goal = self.df_filt

        self.relabel(self.df_goal)

        self.df_labeled = self.df_goal
        if not self.validate:
            self.df_labeled = self.df_labeled.reset_index()

        self.nsucs = 0
        self.ntries = 0

        nfiles = os.stat(self.where2store).st_nlink
        successnames = []
        for i in range(self.df_labeled.shape[0]):
            ang = self.df_labeled.f_ang[i]+30.
            if angles =='B': 
                ang += 180.
            elif angles == 'R':
                ang += 270
            elif angles =='L':
                ang +=90
            self.attempts = 0
            self.df_success = pd.DataFrame()
            self.angles = angles
            self.filename = self.info2name(self.df_labeled.lat[i],self.df_labeled.lng[i], self.angles)
            if os.path.exists(self.where2store+self.filename):
                successnames.append(self.filename)
                self.ntries +=1
                self.nsucs +=1
                print self.nsucs, 'already exists:', self.filename
            else:
                eps = self.dmin/100
                self.success = False
                for self.attempts in range(nattempts):
                    self.single_query(self.df_labeled.lat[i]+eps*(np.random.rand()-1)\
                        ,self.df_labeled.lng[i]+eps*(np.random.rand()-1),ang, self.df_labeled.label[i], i )
                    if self.success:
                        self.nsucs +=1
                        self.ntries +=1
                        successnames.append(self.filename)
                        break
                    elif(self.attempts == (nattempts-1)):
                        successnames.append('')
                        self.ntries +=1
                        print 'could not retrive:  %s' %self.filename
                time.sleep(0.0)
        self.df_labeled['successnames'] = successnames
        nexisting = 0
        self.df_folder= pd.DataFrame()
        if os.path.exists(self.where2store+'folderdata.csv'):
            self.df_folder = self.df_labeled.append(pd.read_csv(self.where2store+'folderdata.csv')\
                [self.df_labeled.columns],ignore_index=True)       
            self.df_folder.drop_duplicates('successnames', inplace=True)
        else:
            self.df_folder = self.df_labeled
        self.df_folder.to_csv(self.where2store+'folderdata.csv')






