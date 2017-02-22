# -*- coding: utf-8 -*-
"""
Created on Sat Feb 04 16:03:13 2017
 
@author: bahareh
"""
 
# -*- coding: utf-8 -*-
 
# Example of loading the data.
 
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
if __name__== '__main__':
 
	data_path = "C:/Users/bahareh/.spyder/Data" # This folder holds the csv files
 
	# load csv files. We use np.loadtxt. Delimiter is ","
	# and the text-only header row will be skipped.
	
    print("Loading data...")
	x_train = np.loadtxt(data_path + os.sep + "x_train.csv",
                         delimiter = ",", skiprows = 1)
	x_test  = np.loadtxt(data_path + os.sep + "x_test.csv",
                         delimiter = ",", skiprows = 1)	
	y_train = np.loadtxt(data_path + os.sep + "y_train.csv",
                         delimiter = ",", skiprows = 1)
	
	print "All files loaded. Preprocessing..."
   # id_x_test =  x_test[:,:1]
 
	# remove the first column(Id)
	x_train = x_train[:,1:]
	x_test  = x_test[:,1:]  
	y_train = y_train[:,1:]
	
 
	# Every 100 rows correspond to one gene.
	# Extract all 100-row-blocks into a list using np.split.
	num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100
 
	print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
	x_train = np.split(x_train, num_genes_train)
	x_test  = np.split(x_test, num_genes_test)
 
	# Reshape by raveling each 100x5 array into a 500-length vector
	x_train = [g.ravel() for g in x_train]
	x_test  = [g.ravel() for g in x_test]
	
	# convert data from list to array
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test  = np.array(x_test)
	y_train = np.ravel(y_train)
	
	# Now x_train should be 15485 x 500 and x_test 3871 x 500.
	# y_train is 15485-long vector.
	
    print("x_train shape is %s" % str(x_train.shape))	
    print("y_train shape is %s" % str(y_train.shape))
	print("x_test shape is %s" % str(x_test.shape))
	
	print('Data preprocessing done...')
	
	print("Next steps FOR YOU:")
    print("-" * 30)
	print("1. Define a classifier using sklearn")
	print("2. Assess its accuracy using cross-validation (optional)")
	print("3. Fine tune the parameters and return to 2 until happy (optional)")
	print("4. Create submission file. Should be similar to y_train.csv.")
	print("5. Submit at kaggle.com and sit back.")

	model = LogisticRegression()
	model.fit(x_train, y_train)
    Pred_probab=model.predict_proba(x_test)
	Pred_Probab = Pred_probab [:,1:]

	f = open('result.csv', 'r+')
    f.write('GeneId,Prediction\n')
	for i in range(0,len(Pred_Probab)):
        f.write("%d,%f\n" % (i+1, Pred_Probab[i]))
	f.close() 