# -*- coding: utf-8 -*-
"""
Created on Sat Feb 04 16:03:13 2017

@author: bahareh, ATATURK
"""

# -*- coding: utf-8 -*-

# Example of loading the data.

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import decomposition, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split


if __name__== '__main__':

    #data_path = "C:/Users/bahareh/.spyder/Data" # This folder holds the csv files
    data_path = "E:/TUT/p34/PatternRecognitionandMachineLearning/Competiton/" # This folder holds the csv files


    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.
    
    print("Loading data...")
    #samples
    x_train = np.loadtxt(data_path + os.sep + "/train/x_train.csv", 
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "/test/x_test.csv", 
                         delimiter = ",", skiprows = 1)    
    #labels
    y_train = np.loadtxt(data_path + os.sep + "/train/y_train.csv", 
                         delimiter = ",", skiprows = 1)
    
    print "All files loaded. Preprocessing..."
   # id_x_test =  x_test[:,:1] 

    # remove the first column(Id)
    x_train = x_train[:,1:] 
    x_test  = x_test[:,1:]   
    y_train = y_train[:,1:] 

#### which one is most important
    
#    index_to_remove = 4
#    x_train = np.delete(x_train, index_to_remove, 1)
#    x_test = np.delete(x_test, index_to_remove, 1)
    
#### end of which one is most important


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
    
#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=(3871/15485.0), random_state=0)
#
#    C = 0.01
#    penalty = "l1"
#    
#    model = LogisticRegression()
#    model.C = C
#    model.penalty = penalty
#    model.fit(X_train, y_train)
#    pred=model.predict(X_test)
#    print(accuracy_score(y_test, pred))

    #5-features: 0.852234564712
    #######without first: 0.835184706794
    #without second: 0.844742960475
    #without third: 0.846292947559
    #without forth: 0.845001291656
    #without fifth: 0.850167915267
    

#### finding best C and penalty combination

#    clf = LogisticRegression()
#    C_range = 10.0 ** np.arange(-6, 1)
##    C_range=np.linspace(0.00001,1)
#    for C in C_range:
#        for penalty in ["l1", "l2"]:
#            clf.C = C
#            clf.penalty = penalty
#            scores = cross_validation.cross_val_score(clf, x_train, y_train)
#            print("C:%f; Penalty:%s; Score:%s" % (C, penalty, str(scores.mean())))


#### using principal component analysis (PCA)

##    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#
#    #clf = LogisticRegression(penalty='l1', C=0.01)    # 0.854698094931
#    clf = LogisticRegression()          # best score: 0.854617, best parameters: {'logistic__C': 1.0, 'pca__n_components': 60}
#    pca = decomposition.PCA()
#    pipe = Pipeline(steps=[('pca', pca), ('logistic', clf)])
#    n_components = range(10,110,10)   ### [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#    #Cs = np.logspace(-4, 4, 3)
#    Cs = 10.0 ** np.arange(-6, 1)
#
#    #Parameters of pipelines can be set using ‘__’ separated parameter names:
#    model = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
#    model.fit(x_train, y_train) 
#    print("best score: %f, best parameters: %s" % (model.best_score_, model.best_params_))
#    print("best estimator: %s" % (model.best_estimator_))
#    ## best score: 0.854617, best parameters: {'logistic__C': 1.0, 'pca__n_components': 60}
#
#    Pred_probab=model.predict_proba(x_test)
#    Pred_Probab = Pred_probab [:,1:]
#  

##q2
#### KNN
#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#
#    for i in range(1,11):        #range(1,11)->1.2.3...10
#        clf = KNeighborsClassifier()  # KNN method; best score is 0.834668044433 with N=7, test_size=(3871/15485.0).
#        clf.n_neighbors=i
#        clf.weights='distance'
#        
#        clf.fit(X_train, y_train) #fitting; teaching
#        y_predictions = clf.predict(X_test)  #prediction
#        print("KNN-%d, %f" % (i, accuracy_score(y_test, y_predictions )))
#        
##KNN-1, 0.761705
##KNN-2, 0.763965
##KNN-3, 0.811753
##KNN-4, 0.811108
##KNN-5, 0.831127
##KNN-6, 0.832096
##KNN-7, 0.837908
##KNN-8, 0.837262
##KNN-9, 0.835002
##KNN-10, 0.836293        

#### RandomForestClassifier
    
#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#    clf = RandomForestClassifier()
#    clf.n_estimators = 700
#    #100-score:0.848563125605; 
#    #200-score:0.849531804973; 
#    #300-score:0.847917339361; 
#    #250-score:0.848240232483; 
#    #190-score:0.848886018728
#    #####210-score:0.851146270584;
#    #220-score:0.848240232483
#    #215-score:0.845979980626
#    #700-score:0.846625766871
#    clf.fit(X_train, y_train)
#    y_pred=clf.predict(X_test)
#    score=accuracy_score(y_test, y_pred)
#    print score

##internet D.I.S.
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    n_est=800
    
    rfc = RandomForestClassifier(n_jobs=-1, max_features= 'log2', n_estimators=n_est, oob_score = True)
    rfc.fit(X_train, Y_train)
    print n_est
    print rfc.oob_score_
    y_pred=rfc.predict(X_test)
    print accuracy_score(Y_test, y_pred)
    print roc_auc_score(Y_test, y_pred)

#    est = range(600,1100,100)
    
#    param_grid = { 
#                  'n_estimators': [800], 
#                }
#    
##    param_grid = { 
##                  'n_estimators': [200, 700],
##        'max_features': ['auto', 'sqrt', 'log2']
##    }
#
#    model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#    model.fit(x_train, y_train)
#    print model.best_params_
#    print model.best_score_     
#    print model.best_estimator_ 
    
    
    
#### cross validation
#    from sklearn.cross_validation import cross_val_score, StratifiedKFold
#    skf = StratifiedKFold(y_train, 10, shuffle = True)
#    scores = cross_val_score(rfc, X_train, y_train, cv = skf)
#    print ("Accuracy: %.2f +- %.2f" % (np.mean(scores), np.std(scores)))
    
#{'max_features': None, 'n_estimators': 800}
#with splitting
#0.852114949952
# took sooooo loong 30min
    
#{'max_features': 'auto', 'n_estimators': 800}
#with splitting
#0.853890862125    

#max_features= 100
#{'n_estimators': 800}
#0.852179528576
#without splitting
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features=100, max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=800, n_jobs=-1, oob_score=True, random_state=None,
#            verbose=0, warm_start=False)
    
    
#{'max_features': 'auto', 'n_estimators': 800}
#without splitting
#0.852373264449
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=800, n_jobs=-1, oob_score=True, random_state=None,
#            verbose=0, warm_start=False)
    

#q3
#### finding best C and penalty combination

#    clf = KNeighborsClassifier()  
#    clf.n_neighbors=7
#    clf.weights='distance'
#    C_range = 10.0 ** np.arange(-3, 1)
#
#    for C in C_range:
#        for penalty in ["l1", "l2"]:
#            clf.C = C
#            clf.penalty = penalty
#            scores = cross_validation.cross_val_score(clf, x_train, y_train)
#            print("C:%f; Penalty:%s; Score:%s" % (C, penalty, str(scores.mean())))
#
##### using best C and penalty combination
#
#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#    clf = KNeighborsClassifier()
#    clf.n_neighbors=7
#    clf.weights='distance'
#    clf.C = 
#    clf.penalty = ""
#    clf.fit(X_train, y_train)
#    pred=model.predict(X_test)
#    accuracy_score(y_test, pred)

#    clf.fit(x_train, y_train)
#    Pred_probab=clf.predict_proba(x_test)
#    Pred_Probab = Pred_probab [:,1:]
    

#q4
#### using principal component analysis (PCA)

#    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

#    clf = KNeighborsClassifier()          # best score: , best parameters: {'logistic__C': 1.0, 'pca__n_components': 60}
#    pca = decomposition.PCA()
#    pipe = Pipeline(steps=[('pca', pca), ('KNN', clf)])
#    n_components = range(10,110,10)   ### [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#    Cs = 10.0 ** np.arange(-6, 1)
#
#    #Parameters of pipelines can be set using ‘__’ separated parameter names:
#    model = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
#    model.fit(x_train, y_train) 
#    print("best score: %f, best parameters: %s" % (model.best_score_, model.best_params_))
#    print("best estimator: %s" % (model.best_estimator_))
#    ## best score: , best parameters: {'logistic__C': 1.0, 'pca__n_components': 60}
#

#    Pred_probab=model.predict_proba(x_test)
#    Pred_Probab = Pred_probab [:,1:]
#            
####### writing file   
#    f = open('result.csv', 'r+')
#    f.write('GeneId,Prediction\n')
#    for i in range(0,len(Pred_Probab)):
#        f.write("%d,%f\n" % (i+1, Pred_Probab[i]))
#    f.close() 
