#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: maxent.py
# @Author: Isaac Caswell/cs224u staff
# @created: 27 May 2015
#
#===============================================================================
# DESCRIPTION:
#
# 
# For example:
#
# The code is largely adapted from cs244u hw4.
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE:
#===============================================================================
from sklearn.svm import SVC
import numpy as np


import json
from collections import Counter
import re
from pprint import pprint
import numpy as np
from distributedwordreps import *
import sklearn.metrics
import operator
import scipy
import util
import time
import csv

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFpr, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

class Maxent:
    def __init__(self,):
        self.name = "Logistic Regression classifier"
        self.model = None

    #=============================================================================================
    # PUBLIC INTERFACE
    # These methods are common to all the classifiers we write


    def train_classifier(self, 
            src_filename,
            feature_function=None,
            feature_selector=None,#SelectFpr(chi2, alpha=0.05), # Use None to stop feature selection
            cv=8, # Number of folds used in cross-validation
            priorlims=np.arange(.1, 4.0, .3), #TODO: these are arbitrary numbers!
            use_rfe = False,
            param_search = False, 
            print_model = False,
            aux_data = None,
            retrain = True): # regularization priors to explore (we expect something around 1)

        if feature_function is None:
            feature_function = self._unigram_ft_fn
        """
        note: this differs from the class implementation in that you pass in the filename to read, not the 
        reader itself.  The advantage of this is that it is less annoying.  The disadvantage is that it's
        less general in the case that you want to use different filetypes.

        @return: a tuple of
                (
                    mod - a trained model capable of prediction,
                    vectorizer - an object to convert a nice Counter to a numeric feature vector,
                    feature_selector - the feature selector to used on the training data/to use on the the test data,
                    feature_function - the function used to featurize the trainging data/to use on the the test data
                )

        meme - a (self replicating,) nongenetic cultural unit

        TODO:
        The following errors arrive from using too many cv folds:
        ValueError: zero-size array to reduction operation maximum which has no identity

        """
        if retrain:
            self.model = None

        reader=util.binarized_transcript_reader(src_filename)
        # Featurize the data:
        feats, labels = self._featurizer(reader=reader, feature_function=feature_function) 
        
        # Map the count dictionaries to a sparse feature matrix:
        vectorizer = DictVectorizer(sparse=True) #TODO this was false in the 224u code. No idea why.

        # X is a list of lists, each of shich have length of about 1000
        X = vectorizer.fit_transform(feats)

        # Define the basic model to use for parameter search:
        searchmod = LogisticRegression(fit_intercept=True, intercept_scaling=1, solver = 'lbfgs')
        
        ##### FEATURE SELECTION    
        # (An optional step; not always productive). By default, we select all
        # the features that pass the chi2 test of association with the
        # class labels at p < 0.05. sklearn.feature_selection has other
        # methods that are worth trying. I've seen particularly good results
        # with the model-based methods, which require some changes to the
        # current code.
        feat_matrix = None
        if use_rfe:
            feature_selector = RFE(estimator = searchmod, n_features_to_select=None, step=1, verbose=0)
        if feature_selector:
            feat_matrix = feature_selector.fit_transform(X, labels)
        else:
            feat_matrix = X
        
        if param_search:
            ##### HYPER-PARAMETER SEARCH
            # Parameters to grid-search over:
            parameters = {'C':priorlims, 'penalty':['l1','l2'], 'multi_class': ['ovr', 'multinomial']} 
            # parameters = {'C':priorlims, 'penalty':['l1'], 'multi_class': ['ovr']}  #TODO: actually take the time to search for good params
            # Cross-validation grid search to find the best hyper-parameters:   
         
            clf = GridSearchCV(searchmod, parameters, cv=cv)
            # import pdb;pdb.set_trace()
            print "searching for optimal hyperparameters..." 
            clf.fit(feat_matrix, labels)
            print "whew, done with that grid search"
            params = clf.best_params_
        else:
            """Best model LogisticRegression(C=3.7000000000000006, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', penalty='l1', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0)"""
            params = {'C':3.7, 'penalty':'l1', 'multi_class': 'ovr'} 

        # Establish the model we want using the parameters obtained from the search:
        mod = LogisticRegression(fit_intercept=True, 
            intercept_scaling=1, 
            C=params['C'], 
            penalty=params['penalty'], 
            multi_class = params['multi_class'], 
            solver = 'lbfgs')

        ##### ASSESSMENT              
        # Cross-validation of our favored model; for other summaries, use different
        # values for scoring: http://scikit-learn.org/dev/modules/model_evaluation.html
        if print_model:
            scores = cross_val_score(mod, feat_matrix, labels, cv=cv, scoring="f1_macro")       
            print 'Best model', mod
            print '%s features selected out of %s total' % (feat_matrix.shape[1], X.shape[1])
            print 'F1 mean: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

        # TRAIN OUR MODEL:
        print "training model..."
        mod.fit(feat_matrix, labels)
        print "done with training, yeah"

        # Return the trained model along with the objects we need to
        # featurize test data in a way that aligns with our training
        # matrix:
        self.model = (mod, vectorizer, feature_selector, feature_function)


    def classify(self, src_filename):
        """predict the labels of each line in src_filename.
        TODO: return the predicted probability as well

        @return: a tuple of 
                (
                transcript, 
                predicted code id,
                predicted code,
                gold_code
                )
        """
        reader=util.binarized_transcript_reader(src_filename)
        mod, vectorizer, feature_selector, feature_function = self.model
        feats, labels = self._featurizer(reader=reader, feature_function=feature_function)
        feat_matrix = vectorizer.transform(feats)
        if feature_selector:
            feat_matrix = feature_selector.transform(feat_matrix)
        predictions = mod.predict(feat_matrix)
        return zip(['transcript excluded']*len(predictions), 
                ['whatever']*len(predictions), 
                predictions, 
                labels) 

    #=============================================================================================
    # INTERNAL WORKINGS/PRIVATE FUNCTIONS
    # yeah

    def _unigram_ft_fn(self, x_i):
        """
        returns a counter.  replace me with a better feature function and put me in featurizer.py
        """
        assert(isinstance(x_i, str))
        return Counter(x_i.split())


    def _featurizer(self, reader, feature_function):
        """Map the data in reader to a list of features according to feature_function,
        and create the gold label vector.

        @param reader: a function which when called returns a generator-like object 
                        yeilding tuples of (label_i, transcript), where transcript is
                        a transcript string and label_i is the gold class label thereof.
            --> the reason it's a generator and not a normal data structure is for scalability!

        @return: tuple(
                    feats:  a list of Counters representing feature vectors returned from
                            feature_function
                    labels: the true labels
                    )
        """
        feats = []
        labels = []
        split_index = None
        for label, x_i in reader:
            d = feature_function(x_i)
            feats.append(d)
            labels.append(label)              
        return (feats, labels)
