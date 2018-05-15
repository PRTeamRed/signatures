#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:34:13 2018

@author: C.B. Doorenbos
"""

import scipy as sc;
import os;
import scipy.spatial.distance;
import pandas as pd;

def DTWDistance (feature_vector_1, feature_vector_2, distancemetric = 'euclidean'):
    """
    Computes dynamic time warping distance
    Arguments:
        feature_vector_1 and 2:
            The vectors for which the DTW distance must be computed. Expected
            format: numpy array where dimension 0 corresponds to the features,
            dimension 1 corresponds to the time domain.
        distancemetric:
            Same options as scipy.spatial.distance.cdist():
            e.g. euclidean (default), cityblock, cosine...

    Returns the DTW distance
    """
    # Compute distance matrix for all sets of feature vectors
    ndim = feature_vector_1.shape[1];
    if ndim != feature_vector_2.shape[1]:
        raise ValueError("Feature vectors must have the same number of columns!");

    N1 = feature_vector_1.shape[0];
    N2 = feature_vector_2.shape[0];
    bandwidth = int(N1/5+1);

    distance_matrix = sc.zeros([N1,N2]);
    N_quotient = float(N1)/float(N2);
    for j in range(N2):
        imin = max(0,  int(j*N_quotient - bandwidth));
        imax = min(N1, int(j*N_quotient + bandwidth) + 1);

        distance_matrix[imin : imax, j] = sc.spatial.distance.cdist(
                feature_vector_1[imin:imax,:],
                feature_vector_2[j,:].reshape(1,ndim) ).flatten();

        if j == 0:
            for i in range(imin+1, imax):
                distance_matrix[i, j] += distance_matrix[i-1, j];
        else:
            if imin == 0:
                distance_matrix[imin, j] += distance_matrix[imin,   j-1];
            else:
                distance_matrix[imin, j] += min([distance_matrix[imin-1, j-1],
                                                 distance_matrix[imin,   j-1]]);

            for i in range(imin+1, imax_previous):
                distance_matrix[i, j] += min([distance_matrix[i-1, j],
                                              distance_matrix[i-1, j-1],
                                              distance_matrix[i,   j-1]]);
            if imax > imax_previous:
                distance_matrix[imax_previous, j ] += \
                        min([distance_matrix[imax_previous-1, j],
                             distance_matrix[imax_previous-1, j-1]]);

                for i in range(imax_previous+1, imax):
                    distance_matrix[i, j] += distance_matrix[i-1, j];

        imax_previous = imax;

    return distance_matrix[-1, -1];

def read_signature_files(directory):
    signatures = {}

    filelist = os.listdir(directory);
    for filename in filelist:

        # check if file is actually a textfile;
        if filename.split(".")[-1] != "txt":
            continue;

        # split filename
        writerID = filename.split(".")[0].split("-")[0];
        n        = filename.split(".")[0].split("-")[-1];

        # for first signature of writer, create dictionary
        if not (writerID in signatures):
            signatures[writerID] = {};

        # load file contents
        signatures[writerID][n] = sc.genfromtxt(
                directory + "/" + filename,
                delimiter = " ");

    return signatures;




def compute_features(data):
    result = sc.zeros([data.shape[0], 6]);
    r = sc.sqrt(data[:,1]**2 + data[:,2]**2);
    r_std = r.std();
    dt = data[1:, 0] - data[:-1, 0];

    result[:, 0] = (data[:,1] - data[:,1].mean()) / r_std;           # x, normalised
    result[:, 1] = (data[:,2] - data[:,2].mean()) / r_std;           # y, normalised
    result[1:,2] = (result[1:, 0] - result[:-1, 0]) / dt;            # vx
    result[1:,3] = (result[1:, 1] - result[:-1, 1]) / dt;            # vy
    result[:, 4] = (data[:,3] - data[:,3].mean()) / data[:,3].std(); # pressure, normalised
    result[:, 5] =  data[:,4];                                       # penup
    return result;


def main():
    train_directory = "./enrollment";
    test_directory  = "./verification";

    # Data structure: train_sigs contains all signatures from the folder enrollment
    # train_sigs[writerID] contains the five signatures of person writerID
    # train_sigs[writerID][n] contains the nth signature of person writerID

    train_sigs = read_signature_files(train_directory);
    test_sigs  = read_signature_files(test_directory);

    for writerID in train_sigs:
        for i in train_sigs[writerID]:
            train_sigs[writerID][i] = compute_features(train_sigs[writerID][i]);
        for i in test_sigs[writerID]:
            test_sigs[writerID][i]  = compute_features(test_sigs[writerID][i]);

    dissimilarities = {};

    for writerID in train_sigs:
        dissimilarities[writerID] = {};
        for i in test_sigs[writerID]:
            dissimilarity = sc.inf;
            for j in train_sigs[writerID]:
                dissimilarity = min(dissimilarity,
                        DTWDistance(test_sigs[writerID][i], train_sigs[writerID][j]));
            dissimilarities[writerID][i] = dissimilarity;
            print ("Writer: "+writerID+", i: "+i+", dissimilarity: {}".format(dissimilarity));



if __name__ == "__main__":
    main();
