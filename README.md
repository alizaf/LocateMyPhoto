# Locate This View

Locate This View is a Convolutional Neural Network algorithm designed and evaluated to estimate the exact location of a street view image from San Francisco.

# Motivations

Street view photos often contain a lot of information about businesses, propoerties, news, etc. An easy way to detect an image's location is to refer to its geo-tag information. However, there are limited number of images which carry a geo-tag inforation attached. Is there a way for us to estimate location of the vast majority of street view images whitout geo-tag information? 
This project is focuse on finding an answer for this question. In addition, if we are able to find a solution for this problem, it means that we are able to identify street view images based on their visual features. This capaility enables us to potential extract many other information from street view images.

# Method
# Data pipeline

Images for training and testing the algorithm is taken from google street view API. I have collected and calculated lat long information of more than 600 streets ranked based on the number of registered businesses in a given street. Latitude and longitude data for intersecitons are extracted from google geolocation API (using intersection of streets as an address parameter). Finally, regularly distributed points are interpolated every 100 ft, and images with 4 different angles are scraped from google streetview API. 

#Convolutional Neaual Network model

The convolutional neaural network model is designed using three convolutional layers following with two fully connected hidden layers. Nolearn, a python package developed based on Lasagne and Theano, is selected to develop the neaural network model. 

# Results
The neuaral network algorithm of LocateThis View is trained using more than 30000 images and tested on a set of streetview images. I have used these test data to record and analyze the predictions of the model at the end of each epoch. Results are presented as graphs and animations, where we clearly observe how the model learns the features of each area and improves its performance. 
Final model results in more than 70% of the points within 1 km radius from the true value (San Francisco is a 10x10 km area). 