import requests
import urllib
from flask import Flask
from flask import request
#import ipdb
import numpy as np
import pandas as pd
import time
import os
from random import randint

def single_query(lat, longit, angle,ii,nsucs,ntries):
    #getfile = urllib.URLopener()
    # print lat,longit
    link='https://maps.googleapis.com/maps/api/streetview?size=300x400&location=%.6f,%.6f&fov=90&heading=%d&pitch=10&key=AIzaSyCsHPCM6nVYgvItOGnmXq17FhvKtjOp44k'%(lat,longit,angle)
    print link
    #     https://maps.googleapis.com/maps/api/streetview?size=400x400&location=40.720032,-73.988354&fov=90&heading=235&pitch=10
    filename = './photodb6/lat%.6f_lng%.6fang%03d_%d.png' %(lat,longit,angle,ii)
    urllib.urlretrieve(link, filename)
    if os.stat(filename).st_size == 3762 or os.stat(filename).st_size==7850:
        os.remove(filename)
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
    #response = requests.get(link)#, params=payload)
    #return response

def collectpics(csvfilename, sw,ne, meshgrid)
    df = pd.read_csv('latlong_distinct.csv')
    df = df.drop(0,axis=0)

    nsucs = 0
    ntries = 0

    SWsf=[37.70339999999999,-122.527]
    NEsf=[37.812,-122.3482]
    SW1 = [37.785087, -122.423623]
    NE1 = [37.811130, -122.382081]
    latlinspace = np.linspace(SW1[0],NE1[0],6)
    lnglinspace = np.linspace(SW1[1],NE1[1],6)
    #latlong=[]
    mesh = np.array(np.meshgrid(latlinspace, lnglinspace))
    SW = SW1#mesh[:,5,5]
    NE = NE1#mesh[:,10,10]
    def isinthebox(lat, longit, SW, NE):
        if (lat < NE[0]) & (lat > SW[0]) & (longit < NE[1]) & (longit > SW[1]):
            return True
        else:
            return
    angles = [45,135,225,315]#, 45, 120, 135, 210, 225, 300, 315]
    nfiles = os.stat('photodb6/').st_nlink
    for i in range(df.shape[0]):#nfiles+1,42867)nfiles+10001):#range(df.shape[0]):
        if isinthebox(df.iloc[i,1],df.iloc[i,2], SW, NE):
            nsucs, ntries = single_query(df.iloc[i,1],df.iloc[i,2],angles[randint(0,3)],i, nsucs, ntries)
            time.sleep(0.0)

# for i in range(len(meshx)): 
#     for j in range(len(meshy)):
#         single_query(meshy[j],meshx[i],angle,i*59+j)
#         time.sleep(0.2)

        # latlong.append([c['geometry']['location'] for c in json1['results']])
#if response.status_code != 200:
#    print 'WARNING', response.status_code
#else:

if __name__ == '__main__':
    app.run(host='', port=8080, debug=True)
