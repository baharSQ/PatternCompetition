
"""
Created on Sat Feb 04 14:58:12 2017

@author: bahareh
"""
import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import imread
from scipy.ndimage.morphology import white_tophat
from scipy.io import loadmat# -*- coding: utf-8 -*
x = []
with open ("C:/Users/bahareh/.spyder/competition/x_train.csv", "r") as myfile:
    data = csv.reader(myfile)
    your_list = list(data)
x = np.array(your_list)
#print data
#x=np.loadtxt(open("C:/Users/bahareh/.spyder/competition/x_train.csv"))
#with open('file.csv', 'rb') as f:
 #   reader = csv.reader(f)
  #  your_list = list(reader)
print your_list[1048576]
