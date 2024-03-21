import src.data_processing.VecRep as VecRep
import numpy as np
# -----Possible options-----

IMAGE_TYPES = ["NbinMmax_ones", "Nbin"]

"""
Nbin: N by N matrix of 1s and 0s
Sample inputs: 2bin, 3bin, 10bin

NbinMmax_ones: N by N matrix of 1s and 0s, with only M percentage of squares being 1s
Sample input: 2bin50max_ones, 5bin40max_ones
"""

IMAGE_FILTERS = ["unique", "Nmax_ones", "one_island"]

"""
one_island: Outputs a set of images such that each image has only one connected island of 1s 
(Diagonals are not connected)

Pmax_ones: Outputs a set of images such that each image has only P or lower percentage of 1s 
Sample input: 60max_ones

unique: Outputs a set of images such that each image is NOT a simple translation of another image 
in the set
"""
IMAGE_PRODUCT_TYPES = ["ncc", "ncc_scaled"]

EMBEDDING_TYPES = ["pencorr_D"]

"""
pencorr_D: Find the nearest correlation matrix using pencorr, subject to the rank constraint.
Then computes embeddings with D dimensions, then normalize the embeddings before output
Sample input: pencorr_20
"""

# -----Variables-----
imageType = "2bin"
filters = ["unique"]
imageProductType = "ncc"
embeddingType = "zero_neg"
overwrite = {"filter": False, "im_prod": False, "estimate": False, "plot": False}

# -----Execution-----
emb = VecRep.get_BF_embeddings(imageType=imageType, filters=filters, imageProductType=imageProductType,
                               embeddingType=embeddingType, overwrite=overwrite)
