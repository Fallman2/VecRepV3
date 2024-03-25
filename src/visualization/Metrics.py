import os
from statistics import mean
from typing import List

from src.data_processing import FilepathUtils
from src.data_processing.EmbeddingFunctions import get_eig_for_symmetric
import numpy as np
from numpy.typing import NDArray


def get_k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k: int) -> float:
    """
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Get the index of the top K elements in the embeddingsDotProducts (DP) array
    Get the index of the top K + x elements in the imageProducts (IP) array, where x is the number of elements in the IP
    array with the same value as the Kth largest element in the IP array
    Find the intersection between the two above arrays
    """
    k = k + 1  # This takes into account that the closest neighbour to the vector is itself

    # Get the index of the k largest elements in each list
    imgProd_max_index = np.argpartition(imageProducts, -k)[-k:]
    embProd_max_index = np.argpartition(embeddingDotProducts, -k)[-k:]
    # Get the kth largest element of the image products array
    kth_element = imageProducts[imgProd_max_index[0]]
    # Get the index of elements with the same value as the kth element
    kth_element_index = np.where(imageProducts == kth_element)
    # Add the kth elements to the set of k closest neighbours for the image products array
    imgProd_max_index = np.union1d(imgProd_max_index, kth_element_index)
    # Get number of neighbours which remain closest
    similar_neighbours = np.intersect1d(imgProd_max_index, embProd_max_index)

    res = len(similar_neighbours) - 1  # Take into account that the closest neighbour is itself
    return res


def get_normed_k_neighbour_score(imageProducts: NDArray, embeddingDotProducts: NDArray, k: int) -> float:
    """
    :param k: Value of k in k neighbour score
    :param imageProducts: 1 by N array of image product scores between an image and all another images
    :param embeddingDotProducts: 1 by N array of dot products with an embedding and all other embeddings
    :return: The k neighbour score, as defined in the readme
    Same process as k neighbour score, except the final result is between 0 and 1
    """
    kNeighScore = get_k_neighbour_score(imageProducts, embeddingDotProducts, k)

    res = float(kNeighScore / k)
    return res

def get_mean_normed_k_neighbour_score(matrixG: NDArray, matrixGprime:NDArray, k: int) -> float:
    kNeighArray = []
    for rowIndex in range(len(matrixG)):
        kNeighArray.append(get_normed_k_neighbour_score(matrixG[rowIndex], matrixGprime[rowIndex], k))
    return mean(kNeighArray)

def get_frob_distance(imageProductMatrix: NDArray, embeddingMatrix: NDArray) -> float:
    """
    :param imageProductMatrix: Image product array to be compared
    :param embeddingMatrix: Embedding matrix to be compared
    :return: The frobenius distance between the two vectors
    """
    diff = imageProductMatrix - embeddingMatrix
    frobNorm = np.linalg.norm(diff)
    return frobNorm



class SamplePlottingData:
    def __init__(self, kNormNeighbourScore, imagesFilepath):
        self.kNormNeighbourScore = kNormNeighbourScore
        self.imagesFilepath = imagesFilepath
        self.aveNormKNeighbourScore = calculate_average_neighbour_scores(kNormNeighbourScore)


def get_specified_ave_k_neighbour_score(aveNormKNeighbourScore, k: int):
    for score in aveNormKNeighbourScore:
        if score["kval"] == k:
            return score["neighbourScore"]
    raise ValueError(str(k) + " is an invalid value of k")

def get_specified_k_neighbour_scores(kNormNeighbourScores, k:int):
    for score in kNormNeighbourScores:
        if score["kval"] == k:
            return score["neighbourScore"]
    raise ValueError(str(k) + " is an invalid value of k")
def calculate_average_neighbour_scores(kNeighbourScores):
    """
    :param kNeighbourScores:
    :return: Takes in an array of kNeighbour scores and returns an array of dictionaries,
    Each dictionary contains the value of k, and the corresponding average k neighbour score
    """
    output = []
    for score in kNeighbourScores:
        output.append({"kval": score["kval"], "neighbourScore": mean(score["neighbourScore"])})
    return output


def get_plotting_data(*, imageProductMatrix, embeddingMatrix,
                      imagesFilepath):
    """
    :return: A plotting data object which can be used for graphs to evaluate if the embeddings are a good estimate.

    """

    initialEigenvalues, eigVec = get_eig_for_symmetric(imageProductMatrix)
    dotProdMatrix = np.matmul(embeddingMatrix.T, embeddingMatrix)
    finalEigenvalues, eigVec = get_eig_for_symmetric(dotProdMatrix)
    frobDistance = get_frob_distance(imageProductMatrix, dotProdMatrix)
    numImages = len(imageProductMatrix[0])

    # Sweep from k=1 to k = numimages/5 by default. If num images is small then sweep from 1 - 2
    kNormNeighbourScores = apply_k_neighbour(imageProductMatrix, dotProdMatrix, 1, max(int(numImages / 5), 2))

    maxDiff = np.max(np.abs(imageProductMatrix - dotProdMatrix))

    output = PlottingData(initialEigenvalues=initialEigenvalues, finalEigenvalues=finalEigenvalues,
                          frobDistance=frobDistance, kNormNeighbourScores=kNormNeighbourScores, numImages=numImages,
                          imagesFilepath=imagesFilepath, maxDiff=maxDiff)
    return output

def get_sample_plotting_data(*, imageProductMatrix, embeddingMatrix,
                      imagesFilepath):
    dotProdMatrix = np.matmul(embeddingMatrix.T, embeddingMatrix)
    numImages = len(imageProductMatrix[0])

    # Sweep from k=1 to k = numimages/5 by default. If num images is small then sweep from 1 - 2
    kNormNeighbourScores = apply_k_neighbour(imageProductMatrix, dotProdMatrix, 1, max(int(numImages / 5), 2))
    output = SamplePlottingData(kNormNeighbourScores, imagesFilepath)
    return output

