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
    Divide the size of the intersection by K
    """
    k = k + 1 # This takes into account that the closest neighbour to the vector is itself


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

    return (len(similar_neighbours) - 1) / (k - 1) # Take into account that the closest neighbour is itself


def get_frob_distance(imageProductMatrix: NDArray, embeddingMatrix: NDArray) -> float:
    """
    :param imageProductMatrix: Image product array to be compared
    :param embeddingMatrix: Embedding matrix to be compared
    :return: The frobenius distance between the two vectors
    """
    diff = imageProductMatrix - embeddingMatrix
    frobNorm = np.linalg.norm(diff)
    return frobNorm


def apply_k_neighbour(imageProductArray: NDArray, embeddingDotProductArray: NDArray, startingK: int,
                      endingK: int) -> NDArray:
    """
    :param endingK: Ending K neighbour score, inclusive
    :param startingK: Starting K neighbour score, inclusive. finds the neighbour score then increments by one until endingK
    :param imageProductArray: Image product array to be compared
    :param embeddingDotProductArray: A^tA, where A is the embedding matrix
    :return: A 3D array which has the k neighbour score for each image, for each value of k from startingK to endingK.
    In the from of: [[startingK, [list of k neigbour scores]], [startingK + 1, [list of k + 1 neigbour scores]] ... ,
    [[endingK, [list of endingK neigbour scores]]]
    """
    if startingK >= endingK:
        raise ValueError("Starting K should be lower than ending K")
    if endingK > len(imageProductArray):
        raise ValueError("Ending K should be less than the number of images")
    output = []
    kVals = range(startingK, endingK + 1)
    for kval in kVals:
        for imageNumber in len(imageProductArray):
            output.append([kval, [
                get_k_neighbour_score(imageProductArray[imageNumber], embeddingDotProductArray[imageNumber], kval)]])
    output = np.array(output)
    return output


class PlottingData:
    def __init__(self, *, initialEigenvalues, finalEigenvalues, frobDistance, kNeighbourScores, numImages):
        self.initialEigenvalues = np.array(initialEigenvalues)
        self.finalEigenvalues = np.array(finalEigenvalues)
        self.frobDistance = frobDistance
        self.aveFrobDistance = frobDistance / (numImages ** 2)
        self.kNeighbourScores = np.array(kNeighbourScores)


def get_plotting_data(imageProductMatrix: NDArray, embeddingMatrix: NDArray):
    """
    :param imageProductMatrix:
    :param embeddingMatrix:
    :return: A PlottingData object which can be used to make graphs
    """
    initialEigenvalues, = get_eig_for_symmetric(imageProductMatrix)
    dotProdMatrix = np.matmul(embeddingMatrix.T, embeddingMatrix)
    finalEigenvalues, = get_eig_for_symmetric(dotProdMatrix)
    frobDistance = get_frob_distance(imageProductMatrix, dotProdMatrix)
    numImages = len(imageProductMatrix[0])
    # Sweep from k=2 to k = numimages/3 by default. If num images is small then sweep to
    kNeighbourScores = get_k_neighbour_score(2, max(int(numImages / 3), 3))
    output = PlottingData(initialEigenvalues=initialEigenvalues, finalEigenvalues=finalEigenvalues,
                          frobDistance=frobDistance, kNeighbourScores=kNeighbourScores, numImages=numImages)
    return output
