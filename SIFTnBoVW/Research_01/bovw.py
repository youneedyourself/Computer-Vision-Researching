import cv2
import numpy as np
from sklearn.cluster import KMeans
import faiss

#def build_codebook(descriptors, n_iterations=20, n_codewords=200):
def build_codebook(descriptors, n_codewords=200):
# descriptors[i]: a list of feature vectors (sift features)
# len(descriptors): number of images
    # 1. Collect all local features
    all_descriptors = []
    for img_descriptors in descriptors:
        # extract specific descriptors within the image
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    # convert to single numpy array
    all_descriptors = np.stack(all_descriptors)
    
    # 2. Cluster descriptors
    #kmeans = KMeans(n_clusters=n_codewords, random_state=0).fit(all_descriptors)
    ncentroids = n_codewords
    niter = 10
    verbose = True
    x = all_descriptors
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)
    
    # 3. Return centroids
#    return kmeans.cluster_centers_
    return kmeans.centroids
    
    
def save_codebook(codebook, file_path):
    import joblib

    k = codebook.shape[0]
    joblib.dump(codebook, "codebook_{}.pkl".format(k), compress=3)
    
    
def load_codebook(file_path):
    import joblib
    codebook = joblib.load(file_path)
    return codebook
    
    
def represent_image_features(image_descriptors, codebook):
# image_descriptors: a list of local features

    from scipy.cluster.vq import vq
    
    # Map each descriptor to the nearest codebook entry
    img_visual_words, distance = vq(image_descriptors, codebook)
    
    # create a frequency vector for each image
    k = codebook.shape[0]
    image_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        image_frequency_vector[word] += 1
    return image_frequency_vector


def get_idf_params(histograms):
# histograms: Histograms (frequency vectors) of training images
    N = histograms.shape[0]
    n_i = np.sum(histograms > 0, axis=0)
    return (N, n_i)


def calculate_tfidf_histograms(image_frequency_vectors, idf_params):
# image_frequency_vectors[i]: Histogram of visual words of the image i-th
    (N, n_i) = idf_params
    histos  = np.zeros(image_frequency_vectors.shape)
    for histo_id in range(image_frequency_vectors.shape[0]):
        n_d  = np.sum(image_frequency_vectors[histo_id])
        for bin_id in range(len(image_frequency_vectors[histo_id])): 
            histos[histo_id, bin_id] = image_frequency_vectors[histo_id, bin_id]/ n_d * np.log(N/n_i[bin_id])
    return histos


def normalize_feature_vectors(feature_vectors, norm_type='l2'):
# norm_type: l1, l2, max
    from sklearn import preprocessing

    norm_feats = preprocessing.normalize(feature_vectors, norm=norm_type)
    return norm_feats
