#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: featurize.py
# @Author: Isaac Caswell
# @created: 18 May 2015
#
#===============================================================================
# DESCRIPTION:
#
# A script to featurize preprocessed tsv data files into something scipy sparse
# matrices full of nice features.
# 
#
#===============================================================================
# CURRENT STATUS: unimplemented
#===============================================================================
# USAGE:
#===============================================================================
# Ideas/Notes:
# --the data are telegraphicaly short
# --gloVe etc. can be used elsewhere or here???? 
# --n-grams
# --sentiment lexicons?
# --pobably no punctuation
#
# --How to deal with n-grams?  pass in dictionary or them?  Induce dictionary from data?
#
#===============================================================================

import re
import json
import scipy.sparse as spr
import util
from collections import Counter
from nltk.corpus import wordnet as wn

#===============================================================================
# def _ft_fname(fname):
# 	g =re.split('\.', fname)
# 	g[-2] += '_ft_%s'%util.time_string()
# 	g[-1] = 'json'
# 	return '.'.join(g)

#===============================================================================
# Constants

# SRC_PATH = "preprocessed_data/train/" 
# DST_PATH = "featurized_data/train/"
# # each corresponds to one tab in the original file
# FILES_TO_PREPROCESS_STEMS = [
# 			"current_industry_train.tsv",
# 			 "MIP_personal_1_train.tsv",
# 			 "MIP_political_1_train.tsv",
# 			 "PK_brown_train.tsv",
# 			 "PK_pelosi_train.tsv",
# 			 "past_industry_train.tsv",
# 			 "terrorists_train.tsv",
# 			 "current_occupation_train.tsv",
# 			 "MIP_personal_2_train.tsv",
# 			 "MIP_political_2_train.tsv",
# 			 "PK_cheney_train.tsv",
# 			 "PK_roberts_train.tsv",
# 			 "past_occupation_train.tsv"
# 			 ]

# FILES_TO_PREPROCESS = [SRC_PATH + f for f in FILES_TO_PREPROCESS_STEMS]

# #deleteme
# #FILES_TO_PREPROCESS = ["raw_data/toy/toy_2.tsv"]

# FILES_TO_WRITE_TO = [DST_PATH + _ft_fname(f) for f in FILES_TO_PREPROCESS_STEMS]

glv = None
GLV_VOCAB = set()
GLV_VEC_DIM = 0

#a mapping from word to glv row idx
glv_mapping = {}
#===============================================================================

# public functions
def feature_function(s):
	"""
	@param string s: a string to be featurized
	"""
	assert(isinstance(s, str))
	#return synonym_fts(s)
	# return unigram_fts(s)
	# return unigram_fts(s) + bigram_fts(s)
	# return binarized_unigram_fts(s) + bigram_fts(s)
	# return bigram_fts(s)
	# return binarized_unigram_fts(s)
	return synonym_fts(s) + bigram_fts(s)


def unigram_fts(s):
    """
    returns a counter.  replace me with a better feature function and put me in featurizer.py
    """
    return Counter(util.str_parse(s))

def bigram_fts(s):
	sl = util.str_parse(s)
	return Counter(_n_grams(sl, 2))

def binarized_unigram_fts(s):
	feature_counter = Counter()
	words = util.str_parse(s)
	for w in words:
		feature_counter[w] = 1
	return feature_counter


def init_glove():
	""" should be called if glove features are used. This function serves the
	the purpose of making the code run faster if glove is not used"""
	glv = build('../distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)
	GLV_VOCAB = set(glv[1])
	GLV_VEC_DIM = glv[0][0].shape[0]

	#a mapping from word to glv row idx
	glv_mapping = {}
	for i, word in enumerate(glv[1]):
		glv_mapping[word] = i

def glove_features(s, use_avg=True, use_diff=True, use_sum=False):
	feature_counter = Counter()

def hypernym_features(s, excessive=False):
	"""TODO: strip stopwords out first!!"""
	words = util.str_parse(s)
	all_hyps = Counter([h for word in words for h in _hypernym_names(word, excessive)])
	return all_hyps

def synonym_fts(s, excessive=False):
	"""TODO: strip stopwords out first!!"""
	words = util.str_parse(s)
	all_syns = [syn for word in words for syn in _synonym_names(word, excessive)]
	return Counter(all_syns)

#===============================================================================
# private functions

def _hypernym_names(word, excessive):
	puppies = wn.synsets(word)
	if excessive:
		return [w for ss in puppies for h in ss.hypernyms() for w in h.lemma_names()]
	else:
		if puppies:
			puppies = puppies[0]
		return [w for ss in puppies for h in ss.hypernyms() for w in h.lemma_names()]		


def _synonym_names(word, excessive):
	if excessive:
		return [name for ss in wn.synsets(word) for name in ss.lemma_names()]
	else:
		#Only return the names from the first synset
		sss = wn.synsets(word)
		if sss: ss = sss[0]; 
		else: return [word]
		return [name for name in ss.lemma_names()]

def _tokenize(s):
	return s.split()

def _n_grams(s, n):
	"""returns a list of all n_grams in the list s
	note: s is a list and not a string
	"""
	assert(n >0)
	return ["__".join(s[i:i+n]) for i in range(len(s)-n)]



if __name__ == "__main__":
	s = "At the turning of the century, I was a lad of five"
	# print hypernym_features(s)
	print synonym_fts(s)	

	# for fname_i, fname_o in zip(FILES_TO_PREPROCESS, FILES_TO_WRITE_TO):
	# 	print fname_i, '-->', fname_o
	# 	with open(fname_i, 'r')  as input_f, open(fname_o, 'w') as output_f:
	# 		for line in input_f:
	# 			output_f.write(featurize(line))
