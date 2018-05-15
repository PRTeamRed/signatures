#!/usr/bin/env python
# DTW
# Written by C.B. Doorenbos (2018)

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
    import scipy as sc;
    # Compute distance matrix for all sets of feature vectors
    import scipy.spatial.distance;

    ndim = feature_vector_1.shape[0];
    if ndim != feature_vector_2.shape[0]:
        raise ValueError("Feature vectors must have the same number of rows!");

    N1 = feature_vector_1.shape[1];
    N2 = feature_vector_2.shape[1];
    bandwidth = int(N1/5+1);

    distance_matrix = sc.zeros([N1,N2]);
    for j in range(N2):
        imin = max(0,  j - bandwidth);
        imax = min(N2, j + bandwidth + 1);

        distance_matrix[imin : imax, j] = sc.spatial.distance.cdist(feature_vector_1[:,imin:imax].transpose(), feature_vector_2[:,j].transpose().reshape([1,ndim])).flatten();

        if j == 0:
            for i in range(imin+1, imax):
                distance_matrix[i, j] += distance_matrix[i-1, j];
            imax_previous = imax;
        else:
            if imin == 0:
                distance_matrix[imin, j] += min([distance_matrix[imin-1, j-1],
                                                 distance_matrix[imin,   j-1]]);
            else:
                distance_matrix[imin, j] += min([distance_matrix[imin-1, j],
                                                 distance_matrix[imin-1, j-1],
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

    return distance_matrix[-1, -1];
