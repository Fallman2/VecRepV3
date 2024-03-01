import math
import re
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from oct2py import octave

from src.helpers.NearestCorrelationMikeC import nearcorr, ExceededMaxIterationsError


class NonPositiveSemidefiniteError(Exception):
    pass


def get_embeddings_negatives_zeroed(matrixG):
    """
    :param matrixG: MatrixG to decompose
    :return: embeddings generated after all negative eigenvalue of the image product matrix are zeroed.
    Embeddings are then normalized
    """
    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)
    eigenvalues[0 > eigenvalues] = 0
    Droot = np.sqrt(np.diag(eigenvalues))

    matrixA = np.matmul(Droot, eigenvectors.T)

    # Normalizing matrix A
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA


def get_embeddings_mikecroucher_nc(matrixG, nDim=None) -> NDArray:
    """
    :param matrixG: Matrix to be decomposed :param nDim: Number of dimensions of vector embeddings in the embedding
    matrix :return: An embedding matrix with dimensions: nDim by len(matrix G) Approximates matrix G to a Positive
    semi-definite matrix using the nearest correlation algorithm programmed in python by mike croucher
    """
    # Check and clean input
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    # Use the NC matrix algo
    try:
        matrixGprime = nearcorr(matrixG, max_iterations=10000)
    except ExceededMaxIterationsError:
        print("No NC matrix found after 10000 iterations")
        raise

    # Decompose the matrix
    matrixA = get_embeddings_mPCA(matrixGprime, nDim)
    return matrixA


def get_embeddings_PenCorr_nc(matrixG: NDArray, nDim=None) -> NDArray:
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :return: An embedding matrix with dimensions: nDim by len(matrix G)
    Uses the PennCorr method to approximate the closest near correlation matrix
    """
    # Check and clean input
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)


    # Use the PennCorr algorithm
    matrixGprime = penCorr(matrixG, nDim)

    # Decompose the matrix
    matrixA = get_embeddings_mPCA(matrixGprime, nDim)
    return matrixA


def penCorr(matrixG: NDArray, nDim: int) -> NDArray:
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :return: A symmetric matrix with the same dimension as matrix G, which is the nearest correlation matrix with
    nDim non-zero eigenvalues

    Uses oct2py to run this matlab code by Prof SUN Defeng. The exact code used is slightly modified version of the
    files in Rank_CaliMat, which replaces the mexeig.c file with a mexeig.m file with similar functionality
    https://www.polyu.edu.hk/ama/profile/dfsun/Rank_CaliMatHdm.zip
    """
    _ = octave.addpath("C:/Users/WShihEe/PycharmProjects/VecRepV3/src/matlab_functions")
    octave.push("n", len(matrixG))
    octave.push("r_rank", nDim)
    octave.push("G", matrixG)

    # For fixed diagonal constraint
    octave.eval("I_e = [1:1:n]'; J_e = I_e; k_e = length(I_e);")

    # To generate the bound e,l & u
    octave.eval("e = ones(n,1);  ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;")

    # Set options
    octave.eval("OPTIONS.tau    = 0; OPTIONS.tolrel = 1.0e-5;")

    # Execute function
    octave.eval("[X,INFOS] = PenCorr(G,ConstrA,r_rank,OPTIONS);", verbose=False)
    matrixGprime = octave.pull("X")
    return matrixGprime


def get_embedding_matrix(imageProductMatrix: NDArray, embeddingType: str, nDim=None):
    """
    :param imageProductMatrix: Image product matrix to generate vectors
    :param embeddingType: Type of method to generate vector embeddings
    :param nDim: Number of dimensions in the vector embeddings.
    If none that means the nDim = length of image product matrix
    :return:
    """

    if embeddingType == "zero_neg":
        embeddingMatrix = get_embeddings_negatives_zeroed(imageProductMatrix)

    elif re.search('zero_[0-9]?[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        embeddingMatrix = get_embeddings_mPCA(imageProductMatrix, nDim)
    elif embeddingType == "nc":
        embeddingMatrix = get_embeddings_mikecroucher_nc(imageProductMatrix)
    elif re.search('nc_[0-9]?[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        embeddingMatrix = get_embeddings_mikecroucher_nc(imageProductMatrix, nDim=nDim)
    elif re.search('pencorr_[0-9]?[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        embeddingMatrix = get_embeddings_PenCorr_nc(imageProductMatrix, nDim=nDim)
    else:
        raise ValueError(embeddingType + " is not a valid embedding type")
    return embeddingMatrix


def get_eig_for_symmetric(matrixG: NDArray) -> (NDArray, NDArray):
    """
    :param matrixG: Symmetric matrix G
    :return: Eigenvalues, transposed eigenvectors for matrix G.

    Checks that the matrix is symmetric then returns the sorted (in descending order) eigenvalues and the transposed eigenvectors
    """
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    # The eigenvectors are already given in the form of a transposed eigenvector matrix, where the rows represent the
    # eigenvectors instead of columns
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    # Reverses the order form ascending to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors.T[::-1].T
    return eigenvalues, eigenvectors


def get_embeddings_mPCA(matrixG: NDArray, nDim=None):
    """
    :param matrixG: Matrix to be decomposed
    :param nDim: Number of dimensions of the vector embedding. If none, then carries out a normal decomposition
    :return: An embedding matrix, with each vector having nDim dimensions.
    Keeps the largest nDim number of eigenvalues and zeros the rest. Followed by the normalization of the embedding matrix
    Effectively applies a modified PCA to the vector embeddings
    Also used as a way to decompose a matrix to get an embedding with reduced dimensions for other methods
    """
    # Check and clean input
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)

    # Checking that the matrix is positive semi-definite
    for eigenvalue in eigenvalues[:nDim]:
        if not (eigenvalue > 0 or math.isclose(eigenvalue, 0, abs_tol=1e-5)):
            raise NonPositiveSemidefiniteError("Even after zeroing smaller eigenvalues, matrix G is not a positive "
                                               "semi-definite matrix. Consider increasing the value of nDim")

    # Zeros negative eigenvalues which are close to zero
    eigenvalues[0 > eigenvalues] = 0
    Droot = np.sqrt(np.diag(eigenvalues))

    # Slices the diagonal matrix to remove smaller eigenvalues
    Drootm = Droot[:nDim, :]

    matrixA = np.matmul(Drootm, eigenvectors.T)

    # Normalizing matrix A
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA


def check_symmetric(a, atol=1e-05):
    return np.allclose(a, a.T, atol=atol)


def check_emb_similarity(matrix1: NDArray, matrix2: NDArray) -> bool:
    """
    :param matrix1: Embedding matrix 1
    :param matrix2: Embedding matrix 2
    :return: If the 2 matrices are a simple rotation of each other (similar to 4dp)
    """
    if matrix1.shape != matrix2.shape:
        return False
    Q1, R1 = np.linalg.qr(matrix1)
    Q2, R2 = np.linalg.qr(matrix2)
    return np.allclose(R1, R2, atol=1e-4)


def is_valid_matrix_g(matrixG: NDArray, nDim) -> (NDArray, int):
    """
    :param matrixG: Square symmetric matrix to check
    :param nDim: nDim value to check
    :return: Checks that matrix G is a symmetric, square matrix. Also checks that nDim
    is positive non-zero and less than the length of matrix G. Raises an error otherwise.
    If it is valid, return a symmetric matrix G and nDim
    """
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    if not all(len(row) == len(matrixG) for row in matrixG):
        raise ValueError("Matrix G must be a square matrix")

    # make the matrix symmetric
    matrixG = (matrixG + matrixG.T) / 2

    maxDim = len(matrixG[0])
    if nDim is None:
        nDim = maxDim
    if nDim <= 0:
        raise ValueError(str(nDim) + " is <= 0. nDim has to be > 0")
    if nDim > maxDim:
        raise ValueError(str(nDim) + " is > " + str(maxDim) + ". nDim has to be < the length of matrixG")

    return matrixG, nDim
