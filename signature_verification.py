#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Python version 3.5.3
# Should run on Python 2.7 but without multithreading
#
#Created on Tue May 15 08:34:13 2018
#@author: C.B. Doorenbos

import scipy as sc;
import pprint
import os;
import scipy.spatial.distance;
import pandas as pd;
#from sklearn.linear_model import LogisticRegression;

#######################
# P A R A M E T E R S #
#######################

train_directory = "./enrollment";
test_directory  = "./verification";

#train_directory = "./testsigs/enrollment";
#test_directory  = "./testsigs/verification";

#ground_truth_filename = "gt.txt"

#######################


def DTWDistance (feature_vector_1, feature_vector_2, relative_bandwidth= 0.2, distancemetric = 'euclidean'):
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
    bandwidth = int(N1*relative_bandwidth/2.+1); # Width of Sakoe Chiba band

    distance_matrix = sc.zeros([N1,N2]);
    N_quotient = float(N1)/float(N2);
    for j in range(N2):
        imin = max(0,  int(j*N_quotient - bandwidth));
        imax = min(N1, int(j*N_quotient + bandwidth) + 1);

        distance_matrix[imin : imax, j] = sc.spatial.distance.cdist(
                feature_vector_1[imin:imax,:],
                feature_vector_2[j,:].reshape(1,ndim), distancemetric).flatten();
        #distance_matrix[imin : imax, j] = sc.spatial.distance.cdist(
                #feature_vector_1[imin:imax,:],
                #feature_vector_2[j,:].reshape(1,ndim), distancemetric).flatten();

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

def read_signature_files(directory, read_which = "all"):
    if not (read_which in {"all", "enrollment", "validation"}):
        raise ValueError ("read_which should be 'all', 'enrollment' or 'validation'.");

    signatures = {}
    filelist = os.listdir(directory);
    for filename in filelist:

        # check if file is actually a textfile;
        if filename.split(".")[-1] != "txt":
            continue;

        if read_which != "all":
            if (read_which == "enrollment") ^ (filename.split(".")[0].split("-")[1] == "g"):
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
    result = sc.zeros([data.shape[0], 12]);
    r = sc.sqrt(data[:,1]**2 + data[:,2]**2);
    r_std = r.std();

    result[:, 0]   = (data[:,1] - data[:,1].mean()) / r_std;      # x, normalised
    result[:, 1]   = (data[:,2] - data[:,2].mean()) / r_std;      # y, normalised
    result[:, 2]   = (r - r.mean()) / r_std;                      # r, normalised
    result[1:,3]   = (result[1:, 0] - result[:-1, 0]);            # vx
    result[1:,4]   = (result[1:, 1] - result[:-1, 1]);            # vy
    result[1:,5]   = (result[1:, 2] - result[:-1, 2]);            # vr
    result[:,3:6]  = (result[:,3:6] - result[:,5].mean())/result[:,5].std();
    result[1:,6]   = (result[1:, 3] - result[:-1, 3]);            # ax
    result[1:,7]   = (result[1:, 4] - result[:-1, 4]);            # ay
    result[1:,8]   = (result[1:, 5] - result[:-1, 5]);            # ar
    result[:,6:9]  = (result[:,6:9] - result[:,8].mean())/result[:,8].std();
    result[:, 9]   = 3*(data[:,3] - data[:,3].mean()) / data[:,3].std(); # pressure, normalised
    result[:, 10]  =  70*data[:,4];                                       # penup
    result[:, 11]  = (data[:, 5] - data[:,5].mean()) / data[:, 5].std();
    #result[:, 8]   = data[:,0] #timestamp
    return result;


def compare_test_train_set(test_sigs, train_sigs, writerID):
    distance_dict = {};
    for i in test_sigs[writerID]:
        dissimilarities = sc.array([]);
        for j in train_sigs[writerID]:
            dissimilarities = sc.append(dissimilarities,
                    [DTWDistance(test_sigs[writerID][i], train_sigs[writerID][j])]);
        distance_dict[i] = dissimilarities;
    return distance_dict;


#################################################################################

def main():
    # Data structure: train_sigs contains all signatures from the folder enrollment
    # train_sigs[writerID] contains the five signatures of person writerID
    # train_sigs[writerID][n] contains the nth signature of person writerID

    train_sigs = read_signature_files(train_directory, "enrollment");
    test_sigs  = read_signature_files(test_directory, "validation");

    n_test_files = 0;
    for writerID in train_sigs:
        for i in train_sigs[writerID]:
            train_sigs[writerID][i] = compute_features(train_sigs[writerID][i]);
        for i in test_sigs[writerID]:
            test_sigs[writerID][i]  = compute_features(test_sigs[writerID][i]);
            n_test_files += 1;



    # Compute relevant DTW distances on test set
    try:
        import multiprocessing as mp;
        n_threads = mp.cpu_count();
        n_threads = min([n_threads, 16]);
        n_threads = max([n_threads, 2]);

        distance_dict = {};
        with mp.Pool(n_threads) as pool:
            for writerID in test_sigs:
                distance_dict[writerID] = pool.apply_async(compare_test_train_set, (test_sigs, train_sigs, writerID));
            pool.close()
            pool.join()
        for writerID in distance_dict:
            distance_dict[writerID] = distance_dict[writerID].get();
    except:
        print("Multiprocessing not supported, falling back to single thread.");
        distance_dict = {};
        for writerID in test_sigs:
                distance_dict[writerID] = compare_test_train_set(test_sigs, train_sigs, writerID);




    # Compute statistics to be fed to classifier

    stats  = pd.DataFrame(columns = ["writer", "signature", "min_dist", "mean_dist", "max_dist", "rms_dist", "min_mean_internal", "max_mean_internal", "verdict", "rms_internal","mean_internal"], index = range(n_test_files));

    stats.loc[:]["min_dist"] = pd.to_numeric(stats.loc[:]["min_dist"]);
    stats.loc[:]["max_dist"] = pd.to_numeric(stats.loc[:]["max_dist"]);
    stats.loc[:]["min_mean_internal"] = pd.to_numeric(stats.loc[:]["min_mean_internal"]);
    stats.loc[:]["max_mean_internal"] = pd.to_numeric(stats.loc[:]["max_mean_internal"]);
    stats.loc[:]["mean_dist"] = pd.to_numeric(stats.loc[:]["mean_dist"]);
    stats.loc[:]["rms_dist"] = pd.to_numeric(stats.loc[:]["mean_dist"]);
    stats.loc[:]["rms_internal"] = pd.to_numeric(stats.loc[:]["rms_internal"]);
    stats.loc[:]["mean_internal"] = pd.to_numeric(stats.loc[:]["mean_internal"]);


    index = 0;
    for writerID in test_sigs:
        for i in test_sigs[writerID]:
            dissimilarities = distance_dict[writerID][i];
            stats.loc[index, ['writer','signature','min_dist','mean_dist', 'max_dist', 'rms_dist']] =  writerID, i, dissimilarities.min(), dissimilarities.mean(), dissimilarities.max(), sc.sqrt((dissimilarities**2).mean());
            index += 1;


    # Compute characteristics of training signatures
    for writerID in train_sigs:
        names = list(train_sigs[writerID].keys());
        distance_matrix = sc.zeros([len(names), len(names)]);
        root_mean_square = 0.;
        mean = 0.;
        for i in range(len(names)-1):
            for j in range(i+1, len(names)):
                distance_matrix[j, i] = distance_matrix[i, j] = \
                        (DTWDistance(train_sigs[writerID][names[i]], train_sigs[writerID][names[j]]));
                root_mean_square += distance_matrix[i, j]**2;
                mean += distance_matrix[i,j];
        root_mean_square /= sc.math.factorial(len(names)-1);
        mean /= sc.math.factorial(len(names)-1);

        stats.loc[stats["writer"] == writerID, "rms_internal"] = sc.sqrt(root_mean_square);
        stats.loc[stats["writer"] == writerID, "mean_internal"] = mean;
        stats.loc[stats["writer"] == writerID, "max_mean_internal"] = \
                (distance_matrix.max(1)).mean();
        sc.fill_diagonal(distance_matrix, sc.inf);
        stats.loc[stats["writer"] == writerID, "min_mean_internal"] = \
                (distance_matrix.min(1)).mean();


    #stats["diff_min"]  = stats["min_dist"]  / stats["min_mean_internal"];
    #stats["diff_max"]  = stats["max_dist"]  / stats["max_mean_internal"];
    #stats["diff_mean"] = stats["mean_dist"] / stats["mean_internal"];
    stats["diff_rms"]  = stats["rms_dist"]  / stats["rms_internal"];

    stats.sort_values(by=["writer","signature"]);
    outputFile = open(os.path.join(".","sig_ver_prediction.txt"),'w')

    for writerID in test_sigs.keys():
        outputFile.write(writerID+', ');
        for sigID in test_sigs[writerID].keys():
            outputFile.write(sigID+', {}, '.format(float(stats.loc[(stats['writer'] == writerID) & (stats['signature'] == sigID)]['diff_rms'])))
        outputFile.write('\n');

    # Monte carlo cross validation to find best distance
    # We found it to be the relative root mean square distance

    #X = sc.array(stats[[ "diff_rms" ]]);
    #from sklearn.neural_network import MLPClassifier;
    #ground_truth_file = open(ground_truth_filename);
    #ground_truth = ground_truth_file.readlines();
    #for line in ground_truth:
    #    writerID = line[:3];
    #    sig = line[4:6];
    #    if line[7] == "g":
    #        status = 1;
    #    else:
    #        status = 0;
    #    stats.loc[(stats["writer"] == writerID) & (stats["signature"] == sig) ,"verdict"] = status;
    #
    #y = sc.array(stats['verdict'], int)
    #model1 = LogisticRegression();
    ##model1 = MLPClassifier(hidden_layer_sizes = (10,20,10),learning_rate = "constant" ,learning_rate_init=.008);
    #
    #s = sc.append(sc.ones(int(n_test_files/2), bool), sc.zeros(int(n_test_files-n_test_files/2), bool));
    #sc.random.seed();
    #cross_val = sc.zeros(100);
    #for i in range(100):
    #    sc.random.shuffle(s);
    #    model1.fit(X[s], y[s]);
    ##print ((model1.predict(X) == y).sum());
    #    cross_val[i] = ((model1.predict(X[~s]) == y[~s]).sum()*2./n_test_files);
    #print (cross_val.mean(), cross_val.std());


if __name__ == "__main__":
    main();
