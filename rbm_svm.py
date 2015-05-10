#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: rbm_svmpy
# @Author: Isaac Caswell
# @created: 9 May 2015
#
#===============================================================================
# DESCRIPTION:
#
# A script to rank emails ny importance.  Uses data from Isaac Caswell's gmail
# correspondance, and uses unsupervised deep learning to get features.  These
# features are fed into SVR for classification.
#
#===============================================================================
# CURRENT STATUS: unimplemented
#===============================================================================
# USAGE:
# 
#===============================================================================
# INPUT:
#
#===============================================================================
# OUTPUT:
#
#===============================================================================
# TODO: 
# MAKE ft. REPRESENTATION SPARSE!!!


#standard modules
import numpy as np
import time
from collections import Counter
import sklearn
import matplotlib.pyplot as plt
import argparse
import util as util
from sklearn.svm import SVC
import string

#===============================================================================
# CONSTANTS
#===============================================================================

#remember that full paths are needed for cron jobs and similar, so might as well
OUTPUT_FOLDER = 'output/'
TRAIN_DATA_FILE = 'data/toy_train.tsv'
DEV_DATA_FILE = 'data/toy_test.tsv'
MAX_WORDLEN = 20


k = 20
ALPHA = .1
UNK = 'UUUNKKK'


#===============================================================================
# FUNCTIONS
#===============================================================================

def parse_to_fts(line):
	return line.translate(string.maketrans("",""), string.punctuation).split()

def featurize(line, vocab, feature_to_index_mapping):
	""" SUPER naive.  dense vector, no tf-idf, for loop, etc.  UNK ignored.
	"""
	fts = parse_to_fts(line)
	res = np.zeros((len(vocab)))
	for ft in fts:
		if ft in feature_to_index_mapping:
			res[feature_to_index_mapping[ft]] += 1
		else:
			res[feature_to_index_mapping[UNK]] += 1
	return res
	


#===============================================================================
# SCRIPT
#===============================================================================

#-------------------------------------------------------------------------------
# PREPROCESSING
vocab = set()
vocab.add(UNK)

X_train_data = []
y_train = []

with open(TRAIN_DATA_FILE, 'r') as f:
	firstline = f.readline().split()
	content_i = firstline.index('content')
	importance_id = firstline.index('importance')

	for line in f:
		line = line.split('\t')
		msg = line[content_i]
		X_train_data.append(msg)
		y_train.append(line[importance_id])
		body = set(parse_to_fts(msg))
		vocab |= set(w for w in body if len(w) < MAX_WORDLEN)

feature_to_index_mapping = {w:i for i, w in enumerate(vocab)}

#-------------------------------------------------------------------------------
# load train and test 
# TODO use generators

X_train = []
for msg in X_train_data:
	X_train.append(featurize(msg, vocab, feature_to_index_mapping))


X_test = []
y_test = []

with open(DEV_DATA_FILE, 'r') as f:
	f.readline()
	for line in f:
		line = line.split('\t')
		msg = line[content_i]
		X_test.append(featurize(msg, vocab, feature_to_index_mapping))
		y_test.append(line[importance_id])

#-------------------------------------------------------------------------------
# 
clf = SVC()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print sklearn.metrics.classification_report(y_test, y_pred)





