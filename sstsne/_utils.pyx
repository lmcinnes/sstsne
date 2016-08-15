from libc cimport math
cimport cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY


cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

@cython.boundscheck(False)
cpdef np.ndarray[np.float32_t, ndim=2] _binary_search_perplexity(
        np.ndarray[np.float32_t, ndim=2] affinities,
        np.ndarray[np.int64_t, ndim=2] neighbors,
        np.ndarray[np.int64_t, ndim=1] labels,
        float label_importance,
        int rep_samples,
        float desired_perplexity,
        int verbose):
    """Binary search for sigmas of conditional Gaussians.

    This approximation reduces the computational complexity from O(N^2) to
    O(uN). See the exact method '_binary_search_perplexity' for more details.

    Parameters
    ----------
    affinities : array-like, shape (n_samples, n_samples)
        Distances between training samples.

    neighbors : array-like, shape (n_samples, K) or None
        Each row contains the indices to the K nearest neigbors. If this
        array is None, then the perplexity is estimated over all data
        not just the nearest neighbors.

    labels : array-like, shape (n_samples,)
        Integer labels for the samples. Unlabelled samples should have label -1.
        
    label_importance : float
        Relative importance to place on labelling
        
    rep_sample : int
        Whether the partial labels are a representative sample of the full labelling

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : array, shape (n_samples, n_samples)
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_samples = affinities.shape[0]
    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros((n_samples, n_samples),
                                                       dtype=np.float64)
    # Precisions of conditional Gaussian distrubutions
    cdef float beta
    cdef float beta_min
    cdef float beta_max
    cdef float beta_sum = 0.0
    # Now we go to log scale
    cdef float desired_entropy = math.log(desired_perplexity)
    cdef float entropy_diff

    cdef float entropy
    cdef float sum_Pi
    cdef float sum_disti_Pi
    cdef float prior_prob
    
    cdef long i, j, k, l = 0
    cdef long K = n_samples
    cdef int using_neighbors = neighbors is not None

    cdef np.ndarray[long, ndim=1] label_sizes = np.bincount(labels + 1)
    cdef long n_same_label
    cdef long n_other_label
    cdef long n_unlabelled = label_sizes[0]

    if using_neighbors:
        K = neighbors.shape[1]

    for i in range(n_samples):
        beta_min = -NPY_INFINITY
        beta_max = NPY_INFINITY
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            if using_neighbors:
                for k in range(K):
                    j = neighbors[i, k]
                    P[i, j] = math.exp(-affinities[i, j] * beta)
            else:
                for j in range(K):
                    P[i, j] = math.exp(-affinities[i, j] * beta)
            P[i, i] = 0.0
            sum_Pi = 0.0
            if using_neighbors:
                for k in range(K):
                    j = neighbors[i, k]
                    sum_Pi += P[i, j]
            else:
                for j in range(K):
                    sum_Pi += P[i, j]
            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0
            if using_neighbors:
                for k in range(K):
                    j = neighbors[i, k]
                    P[i, j] /= sum_Pi
                    sum_disti_Pi += affinities[i, j] * P[i, j]
            else:
                for j in range(K):
                    P[i, j] /= sum_Pi
                    sum_disti_Pi += affinities[i, j] * P[i, j]
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == NPY_INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -NPY_INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[t-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))

    if verbose:
        print("[t-SNE] Mean sigma: %f"
              % np.mean(math.sqrt(n_samples / beta_sum)))
              
    for i in range(n_samples):
    
        sum_Pi = 0
        
        if using_neighbors:
        
            for k in range(K):
                j = neighbors[i, k]

                n_same_label = label_sizes[labels[i] + 1]
                n_other_label = n_samples - n_same_label - n_unlabelled
                    
                if rep_samples:
                    
                    denominator = n_same_label ** 2 + n_other_label ** 2 + n_unlabelled ** 2

                    if labels[i] == -1 or labels[j] == -1:
                        prior_prob = n_unlabelled / denominator
                    elif labels[j] == labels[i]:
                        prior_prob = min((n_same_label / denominator) + (label_importance / n_same_label), 1.0 - EPSILON_DBL)
                    else:
                        prior_prob = max((n_other_label / denominator) - (label_importance / n_other_label), EPSILON_DBL)

                else:
                    
                    if labels[i] == -1 or labels[j] == -1:
                        prior_prob = 1.0 / n_samples
                    elif labels[j] == labels[i]:
                        prior_prob = min((1.0 / n_samples) + (label_importance / n_same_label), 1.0 - EPSILON_DBL)
                    else:
                        prior_prob = max((1.0 / n_samples) - (label_importance / n_other_label), EPSILON_DBL)

                P[i, j] *= prior_prob
                sum_Pi += P[i, j]
            
            for k in range(K):
                j = neighbors[i, k]
                P[i, j] /= sum_Pi
                
        else:
        
            for j in range(K):
                
                n_same_label = label_sizes[labels[i] + 1]
                n_other_label = n_samples - n_same_label - n_unlabelled
                    
                if rep_samples:
                    
                    denominator = n_same_label ** 2 + n_other_label ** 2 + n_unlabelled ** 2

                    if labels[i] == -1 or labels[j] == -1:
                        prior_prob = n_unlabelled / denominator
                    elif labels[j] == labels[i]:
                        prior_prob = min((n_same_label / denominator) + (label_importance / n_same_label), 1.0 - EPSILON_DBL)
                    else:
                        prior_prob = max((n_other_label / denominator) - (label_importance / n_other_label), EPSILON_DBL)

                else:
                    
                    if labels[i] == -1 or labels[j] == -1:
                        prior_prob = 1.0 / n_samples
                    elif labels[j] == labels[i]:
                        prior_prob = min((1.0 / n_samples) + (label_importance / n_same_label), 1.0 - EPSILON_DBL)
                    else:
                        prior_prob = max((1.0 / n_samples) - (label_importance / n_other_label), EPSILON_DBL)

                P[i, j] *= prior_prob
                sum_Pi += P[i, j]
            
            for j in range(K):
                P[i, j] /= sum_Pi

    return P
