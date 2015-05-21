#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: generic_util.py
# @Author: Isaac Caswell
# @created: 21 February 2015
#
#===============================================================================
# DESCRIPTION:
#
# A file containing various useful functions I often find I have to write in my 
# scripts/have to look up from other files.  For instance, how to plot things 
# with legends, print things in color, and get command line arguments
#
#===============================================================================
# CURRENT STATUS: Works!  In progress.
#===============================================================================
# USAGE:
# import generic_util as gu
# gu.colorprint("This text is flashing in some terminals!!", "flashing")
# 
#===============================================================================
# CONTAINS:
# 
#-------------------------------------------------------------------------------
# COSMETIC:
#-------------------------------------------------------------------------------
# colorprint: prints the given text in the given color
# time_string:
#       returns a string representing the date in the form '12-Jul-2013' etc.
#       Handy naming of files.
#-------------------------------------------------------------------------------
# FOR (LARGE) FILES:
#-------------------------------------------------------------------------------
# randomly_sample_file: given the name of some unnecessarily large file that you
#       have to work with, original_fname, randomly samples it to have a given
#       number of lines.  This function is used for when you want to do some 
#       testing of your script on a pared down file first.
# scramble_file_lines:
#       randomly permutes the lines in the input file.  If the input 
#       file is a list, permutes all lines in the iput files in the asme way.
#       Useful if you are doing SGD, for instance.
# file_generator:
#       streams a file line by line, and processes that line as a list of integers.
# split_file: given the name of some unnecessarily large file that you have to 
#       work with, original_fname, this function splits it into a bunch of
#       smaller files that you can then do multithreaded operations on.
#
#===============================================================================
# TODO: 
# make general plotting function


#standard modules
import numpy as np
import time
from collections import Counter
import heapq
import matplotlib.pyplot as plt
import argparse


#===============================================================================
# FUNCTIONS
#===============================================================================


def split_file(original_fname, output_dir_fname, n_splits = 15, delimitor = '\n'):
    """given the name of some unnecessarily large file that you have to work with, original_fname,
    this function splits it into a bunch of smaller files that you can then do multithreaded 
    operations on.

    Usage: split_file('./data/words_stream.txt', './data/words_stream_split_15')

    """
    if not os.path.exists(output_dir_fname):
        os.makedirs(output_dir_fname)

    lines_in_file = 0
    with open(original_fname, 'r') as f:
        for line in f:
            lines_in_file += 1

    lines_per_subfile = lines_in_file/n_splits

    with open(original_fname, 'r') as input_f:
        cur_split = 0
        output_f = open(output_dir_fname + '/split_%s'%cur_split, 'w')
        for i, line in enumerate(input_f):
            if i%lines_per_subfile == 0 and i:
                cur_split += 1
                output_f.close()
                output_f = open(output_dir_fname + '/split_%s'%cur_split, 'w')
            output_f.write(line)

        output_f.close()


def make_dev_train_sets(original_fname, names, percents, scramble = False):
    """
    """
    assert(abs(sum(percents) - 1) < 1e-5)
    assert(len(names) == len(percents))

    print 'hello'

    if scramble:
        scramble_file_lines(original_fname, original_fname  + '.scrambled')
        original_fname = original_fname  + '.scrambled'

    lines_in_file = 0
    with open(original_fname, 'r') as f:
        for line in f:
            lines_in_file += 1

    lines_per_split = [p*lines_in_file for p in percents]
    lines_per_split = [np.ceil(n) for n in lines_per_split]

    with open(original_fname, 'r') as input_f:
        cur_split = 0
        output_f = open(names[0], 'w')
        i = 0
        for line in input_f:
            if i%lines_per_split[cur_split] == 0 and i:
                cur_split += 1
                i=0
                output_f.close()
                output_f = open(names[cur_split], 'w')
            output_f.write(line)
            i += 1

        output_f.close()


def randomly_sample_file(original_fname, output_fname, n_lines_to_output = 100, delimitor = '\n', preserve_first_line = 0):
    """given the name of some unnecessarily large file that you have to work with, original_fname,
    randomly samples it to have n_lines_to_output.  This function is used for when you want to
    do some testing of your script on a pared down file first.

    if original_fname is a list, then all files are sampled evenly  (the same lines are taken from 
    each one.)

    Usage: 
    randomly_sample_file(["./data/features.txt", "./data/target.txt"], ["./data/dev_features.txt", "./data/dev_target.txt"], 200)

    randomly_sample_file("data/clean_mail.tsv", "data/toy.tsv")

    """
    assert(type(original_fname) == type(output_fname))
    if isinstance(original_fname, list):
        assert(len(original_fname) == len(output_fname))
    else:
        original_fname = [original_fname]
        output_fname = [output_fname]


    lines_in_file = 0
    with open(original_fname[0], 'r') as f:
        for line in f:
            lines_in_file += 1

    line_idxs_to_output = range(lines_in_file)[1:] if preserve_first_line else np.arange(lines_in_file)
    np.random.shuffle(line_idxs_to_output)
    if preserve_first_line: line_idxs_to_output = [0] + line_idxs_to_output
    line_idxs_to_output = set(line_idxs_to_output[0:n_lines_to_output])

    for input_fname_i, output_fname_i in zip(original_fname, output_fname): 
        with open(input_fname_i, 'r') as input_i, open(output_fname_i, 'w') as output_i:
            for j, line in enumerate(input_i):
                if j in line_idxs_to_output:
                    output_i.write(line)


def scramble_file_lines(original_fname, output_fname, delimitor = '\n'):
    """randomly permutes the lines in the input file.  If the input 
    file is a list, permutes all lines in the iput files in the same way.
    Useful if you are doing SGD, for instance.

    Usage: 
    scramble_file_lines([X_FILENAME, Y_FILENAME], ["./data/scrambled_features.txt", "./data/scrambled_target.txt"])

    """
    assert(type(original_fname) == type(output_fname))
    if isinstance(original_fname, list):
        assert(len(original_fname) == len(output_fname))
    else:
        original_fname = [original_fname]
        output_fname = [output_fname]


    lines = [] #lines[i] is a list of lines
    for input_i in original_fname:
        with open(input_i, 'r') as f:
            for i, line in enumerate(f):
                if len(lines) == i:
                    lines.append([])
                lines[i].append(line)

    np.random.shuffle(lines)

    for i, output_fname_i in enumerate(output_fname): 
        with open(output_fname_i, 'w') as output_i:
            for line in lines:
                output_i.write(line[i])



def file_generator(fname):
    """streams a file line by line, and processes that line as a list of integers."""
    with open(fname, 'r') as f:
        for line in f:
            yield [int(v) for v in line.split()]

def colorprint(message, color="rand"):
    """prints your message in pretty colors! So far, only a few color are available."""
    if color == 'none': print message
    if color == 'demo':
        for i in range(99):
            print '%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
    print '\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'magenta_highlight' : 45,
        'red_highlight' : 41,
        'pink' : 35,
        'yellow' : 93,   
        'teal' : 96,     
        'rand' : np.random.randint(1,99),
        'green?' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m'


def time_string(precision='day'):
    """ returns a string representing the date in the form '12-Jul-2013' etc.
    intended use: handy naming of files.
    """
    t = time.asctime()
    precision_bound = 10 #precision == 'day'
    yrbd = 19
    if precision == 'minute':
        precision_bound = 16
    elif precision == 'second':
        precision_bound = 19
    elif precision == 'year':
        precision_bound = 0
        yrbd = 20
    t = t[4:precision_bound] + t[yrbd:24]
    t = t.replace(' ', '-')
    return t

def convert_to_png(denoised_image):
    """TODO: this is not a real function"""
    plt.imshow(denoised_image, cmap=plt.cm.gray)
    plt.show()



#===============================================================================
# TESTING SCRIPT
#===============================================================================

#-------------------------------------------------------------------------------
# part (i) 

if __name__ == '__main__':
    print 'You are running this from the command line, so you must be demoing it!'
    start_time = time.time()
    #make_dev_train_sets('data/clean_mail.tsv', ['clean_mail_train.tsv', 'clean_mail_dev.tsv', 'clean_mail_test.tsv'], [.7, .2, .1], scramble = True)
    randomly_sample_file("data/clean_mail.tsv", "data/toy_train.tsv", preserve_first_line = 1)
    randomly_sample_file("data/clean_mail.tsv", "data/toy_test.tsv", preserve_first_line = 1)
    k = 10
    domain = range(k)
    f_1 = [i**2 for i in domain]
    f_2 = [np.sin(i) for i in domain]
    f_3 = [np.random.rand()*i for i in domain]


    colorprint("time to do computation: %s"%(time.time() - start_time))
    colorprint(time_string())







