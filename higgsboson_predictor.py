'''
Created on Jul 24, 2014

@author: Jim D'Souza

Description : Code for a Kaggle competition - the Higgs Boson challenge
Attempts to identify a Higgs Boson particle using the data provided.
Uses Random Forests and Gradient Boosting Trees as classifiers.
The data is initially cleaned, scaled, and variables are shortlisted using PCA
'''


### Import relevant libraries ###
import datetime

import csv
import os
import gc
import math

import numpy
from numpy import linalg


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

### Import training and testing data sets  ###
training_file = "D:\\Kaggle\\HiggsBoson\\Data\\Input\\training.csv"
test_file = "D:\\Kaggle\\HiggsBoson\\Data\\Input\\test.csv"

output_file = "D:\\Kaggle\\HiggsBoson\\Data\\Output\\output.csv"


### Function that calculates the AMS of the model being used - this is the testing metric used in the competition
def AMS(output):
     
    # This are the final signal and background predictions
    Yhat = []
    Y_train = []
    W_train = []
    rc= 0
    for row in output :
        if rc > 0 :
            W_train.append(float(row[4]))
            if row[2] == "s" :
                Yhat.append(float(1.0))
            elif row[2] == "b" :
                Yhat.append(float(0.0))
            if row[3] == "s" :
                Y_train.append(float(1.0))
            elif row[3] == "b" :
                Y_train.append(float(0.0))
        rc = rc + 1
    
    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    TruePositive = []
    TrueNegative = []
    for rc in range(len(Y_train)) :
        TruePositive.append(W_train[rc]*(Y_train[rc])*(1.0/0.8))
        TrueNegative.append(W_train[rc]*(Y_train[rc])*(1.0/0.8))
 
    # s and b for the training 
    s_train = 0
    b_train = 0
    for rc in range(len(Yhat)) :
        s_train = s_train + (TruePositive[rc]*Yhat[rc])
        b_train = b_train + (TrueNegative[rc]*Yhat[rc])
 
    # Now calculate the AMS scores
    def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)


def variable_selection(training_feature, training_output):
    
    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 25

    print("Extracting the top %d eigenfaces from %d faces"% (n_components, training_feature.shape[0]))
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(training_feature)
    
    print (pca.explained_variance_ratio_)
    
    print("Projecting the input data on the eigenfaces orthonormal basis")

    training_feature_pca = pca.transform(training_feature)
    print training_feature[0]
    print training_feature_pca[0]
    print pca
    return training_feature_pca, pca


### Creates a random forest model using the sklearn package ###
### Also identifies the feature importances of variables ###
def random_forest(training_header, training_feature,training_output):

    gc.collect()
    
    
    print('Creating the random forest...')
    rf = RandomForestClassifier(n_estimators=20,criterion="entropy",n_jobs=1,compute_importances=True)
    #rf = DecisionTreeClassifier(criterion="entropy",compute_importances=True)

    # fit the training data
    print('Fitting the model...')
    rf.fit(training_feature, training_output) 
    
    # Calculate the feature ranking
    print "Calculating feature importance..."
    
    important_features = []
    for x,i in enumerate(rf.feature_importances_):
        if i>numpy.average(rf.feature_importances_):
            important_features.append(str(x))
        
    importances = rf.feature_importances_
    indices = numpy.argsort(importances)[::-1]
    
    print("Feature ranking:")
    important_names = []
    feature_list = []
    for f in range(len(training_header)):
        important_names.append(training_header[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, training_header[indices[f]], importances[indices[f]])), " : ", indices[f]
        #if f<20 : 
        #    feature_list.append(indices[f])
        #elif not training_header[indices[f]].startswith("DER") :
        #    feature_list.append(indices[f])
        feature_list.append(indices[f])
        
    training_feature_new = training_feature[:,feature_list]
    training_header_new = training_header[feature_list]
    
    del rf
    del important_names
    del importances
    
    return training_header_new, training_feature_new, feature_list

### Function to standardize the data ###
### Steps are described below ###
def standardize_data(feature):    
    # 1) First calculate the mean and standard deviation of all features 
    
    len_feature =  len(feature[0])
    
    print "Calculating mean..."
    feature_mean = numpy.average(feature,axis=0,weights=(feature>-999))
    
    print "Calculating standard deviations..."
    mask = feature > -999
    feature_std = numpy.ma.masked_where(~mask, feature).std(axis=0)
    
    
    # 2) Now replace the masked values with a value in the probability distribution of the variable
    #print "Replacing blank values..."
    #for col in range(len_feature) :
    #    feature[feature[:,col] == -999, col] = feature_mean[col]
        
    # 3) Now replace extreme values, i.e. variables having a value greater than mean +- 2*std dev
    print "Replacing extreme values..."
    for col in range(len_feature) :        
        feature[feature[:,col] > feature_mean[col] + 2*feature_std[col], col] = feature_mean[col] + 2*feature_std[col]
    
    # 4) Standard Scaler - This transforms the features data into mean 0 and variance 1 i.e. Standard Normal form
    feature = StandardScaler().fit_transform(feature)
    
    return feature

### Function to import data ###
def import_data(file):
    in_file = csv.reader(open(file, "rb"))
    x=list(in_file)
    data = numpy.array(x)
    
    return data

### Main function ####
def main(validation):
    
	### Keep track of the running time ###
    start_time = datetime.datetime.now()
    print start_time
    
    ###############################################################################
    # Import training data set
    print "Importing training data set..."
    training_data = import_data(training_file)
    
    training_header = training_data[0,1:31]
    training_data = training_data[1:,:]
    
	### In validation phase, split the given data set into training and testing ###
    if validation == True :
        import random
        random.shuffle(training_data)
        # Creating sample 
        test_data = training_data[200000:,:]
        training_data = training_data[:200000,:]
    
    print "Import time : ", datetime.datetime.now() - start_time
    
    ### Split the training data set into Events, Features and Labels ###
    print "Splitting training data set..."
    training_event = training_data[:,0]
    training_feature = training_data[:,1:31].astype('float')
    training_weight = training_data[:,31]
    training_label = training_data[:,32]
        
    training_output = []
    
    for i in range(len(training_label)) :
        if training_label[i] == "s":
            training_output.append(1)
        else :
            training_output.append(0)
    
    del training_data
    
    ###############################################################################
    # Standardize the training data set
    #training_feature[training_feature ==-999] = 0
    #training_feature = standardize_data(training_feature)

    print "Preprocessing time : ", datetime.datetime.now() - start_time
    
    ###############################################################################
    # Feature selection
    # This is done by using Random Forest to create a number of decision trees...
    # from which we can calculate the importance of each feature.
    # Any feature that has an importance of greater than 3% is selected.
    # In this case, we have selected 9 features from a total of 30
    
    training_header_new, training_feature_new, feature_list = random_forest(training_header, training_feature,training_output)
    #training_feature_new, feature_list = variable_selection(training_feature, training_output)
    
    
    print "Feature Selection completed  : ", datetime.datetime.now() - start_time
    
    ###############################################################################
    # Training the model using Gradient Boosting Tree Classifier
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=10,min_samples_leaf=2000,max_features=20,verbose=1)
    w = clf.fit(training_feature_new, training_output)
    
    # Get the probability output from the trained method, using the 10% for testing
    prob_predict_train = w.predict_proba(training_feature_new)[:,1]
    pcut = numpy.percentile(prob_predict_train,85)
    del prob_predict_train
    
    print "Gradient Boosting Classifier Running time  : ", datetime.datetime.now() - start_time
    
    ###############################################################################
    ### Testing the model using SVM ###
    
    del training_feature
    del training_feature_new
    del training_output
    del training_header
    #del training_header_new
    del clf
    
    gc.collect()

    # Import test data set
    print "Importing test data set..."
    if validation == False :
        test_data = numpy.loadtxt( test_file, delimiter=',', skiprows=1 )
        
    # Get the features from the testing data set
    test_event = test_data[:,0]
    test_feature = test_data[:,1:31].astype('float')
    test_jetnum = test_feature[:,22]
    if validation == True :
        test_label = test_data[:,31:]
    
    # Keep only the most important variables 
    test_feature = test_feature[:,feature_list]
    #test_feature = feature_list.transform(test_feature)
    
    # Standardize the testing feature data
    print "Standardizing data : "
    #test_feature = standardize_data(test_feature)    
    
    print "Predicting output..."
    op = w.predict(test_feature)
    op_w = w.predict_proba(test_feature)
    
    if validation == True :
        test_output = [["EventID","Weight","Class","OP","OP_Weight","Jetnum","RankOrder"]]
    else :
        test_output = [["EventID","Weight","Class","Jetnum","RankOrder"]]
    for row in range(len(test_event)) :
        if op_w[row][1] >= pcut :
            op_v = "s"
        else :
            op_v = "b"
        if validation == True :
            val_row = [test_event[row],op_w[row][1],op_v,test_label[row][1],test_label[row][0],test_jetnum[row]]
        elif validation == False :
            val_row = [test_event[row],op_w[row][1],op_v,test_jetnum[row]]
        test_output.append(val_row)
        
    print "Calculated output in  : ", datetime.datetime.now() - start_time
        
    # Rank by output
    test_output = sorted(test_output, key = lambda x:x[1], reverse=True)
    
    for x in range(len(test_output)-1):
        y = x+1
        test_output[y].append(y)
        
    print "Calculated rank in  : ", datetime.datetime.now() - start_time
         
    with open(output_file, 'wb') as output: 
        writer = csv.writer(output, delimiter=',')
        writer.writerows(test_output)
        
    print "Exported output file  : ", datetime.datetime.now() - start_time
    
    if validation == True :
        AMS(test_output)


if __name__ == '__main__':
    validation = True
    main(validation)