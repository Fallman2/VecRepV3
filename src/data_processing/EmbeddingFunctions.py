import math
import re

import numpy as np
from numpy.typing import NDArray
from oct2py import octave
from sklearn.preprocessing import normalize

from src.helpers.FilepathUtils import get_matlab_dirpath

import src.data_processing.Utilities as utils
from src.data_processing.ImageGenerators import get_triangle_image_set
import src.helpers.FilepathUtils as fpUtils

class NonPositiveSemidefiniteError(Exception):
    pass


def pencorr(matrixG: NDArray, nDim: int) -> NDArray:
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :return: matrix G', a symmetric matrix with the same dimension as matrix G, which is the nearest correlation matrix with
    nDim non-zero eigenvalues

    Uses oct2py to run this matlab code by Prof SUN Defeng. The exact code used is slightly modified version of the
    files in Rank_CaliMat, which replaces the mexeig.c file with a mexeig.m file with similar functionality
    https://www.polyu.edu.hk/ama/profile/dfsun/Rank_CaliMatHdm.zip
    """
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    # TODO what if solver fails
    matlabDir = get_matlab_dirpath()
    _ = octave.addpath(matlabDir)
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

def pencorr_weighted(matrixG: NDArray, nDim: int, matrixH: NDArray) -> NDArray:
    """
    :param matrixG: Symmetric square matrix
    :param nDim: Number of non-zero eigenvalues in the output matrix
    :param matrixH: Weight Matrix
    :return: matrix G', a symmetric matrix with the same dimension as matrix G, which is the nearest correlation matrix with
    nDim non-zero eigenvalues

    Weighted version of the above code included in the same link.
    """
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    matlabDir = get_matlab_dirpath()
    _ = octave.addpath(matlabDir)
    octave.push("n", len(matrixG))
    octave.push("r_rank", nDim)
    octave.push("G", matrixG)
    octave.push("H", matrixH)

    # For fixed diagonal constraint
    octave.eval("I_e = [1:1:n]'; J_e = I_e; k_e = length(I_e);")

    # To generate the bound e,l & u
    octave.eval("e = ones(n,1);  ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;")

    # Set options
    octave.eval("OPTIONS.tau    = 0; OPTIONS.tolrel = 1.0e-5;")

    # Execute function
    octave.eval("[X,INFOS] = PenCorr_HnormMajorDiag(G,H,ConstrA,r_rank,OPTIONS);", verbose=False)
    matrixGprime = octave.pull("X")
    return matrixGprime

def eigencorr(matrixG: NDArray, nDim: int):
    """
    WIP
    :param matrixG:
    :param nDim:
    :return:
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    selected_eigenvalue = eigenvalues[len(matrixG) - nDim]
    corr = matrixG - np.identity(len(matrixG)) * selected_eigenvalue

    return corr, abs(selected_eigenvalue)

def testcorr(matrixG: NDArray, nDim: int):
    """
    Testing method
    :param matrixG:
    :param index:
    :return:
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    selected_eigenvalue = eigenvalues[len(matrixG) - nDim]
    corr = matrixG - np.identity(len(matrixG)) * selected_eigenvalue

    return corr, abs(selected_eigenvalue)

def testcorr100(matrixG: NDArray, nDim: int):
    """
    Testing method
    :param matrixG:
    :param index:
    :return:
    """
    corr = matrixG + np.identity(len(matrixG)) * 100

    return corr, 101

def dblcorr(matrixG:NDArray, nDim: int, weight=None):
    corr, radius = eigencorr(matrixG, nDim)
    if weight is not None:
        matrixGprime = pencorr_weighted(corr, nDim, weight)
    else:
        matrixGprime = pencorr(corr, nDim)
    return matrixGprime

def get_embedding_matrix(imageProductMatrix: NDArray, embeddingType: str, weightMatrix=None):
    """
    :param imageProductMatrix: Image product matrix to generate vectors
    :param embeddingType: Type of method to generate vector embeddings
    :param nDim: Number of dimensions in the vector embeddings.
    If none that means the nDim = length of image product matrix
    :return:
    """
    if re.search(r'pencorr_[0-9]*[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        default_weight = weightMatrix is None or np.array_equal(weightMatrix, np.ones_like(imageProductMatrix))
        if not default_weight:
            matrixGprime = pencorr_weighted(imageProductMatrix, nDim, weightMatrix)
        else:
            matrixGprime = pencorr(imageProductMatrix, nDim)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, nDim)
    elif re.search(r'eigencorr_[0-9]*[0-9]$', embeddingType) is not None:
        nDim = int(re.search(r'\d+', embeddingType).group())
        matrixGprime, radius = eigencorr(imageProductMatrix, nDim)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, nDim, r=radius)
    elif re.search(r'testcorr_[0-9]*[0-9]$', embeddingType) is not None:
        index = int(re.search(r'\d+', embeddingType).group())
        matrixGprime, radius = testcorr(imageProductMatrix, index)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, index, r=radius, abs_tol=100)
    elif re.search(r'testcorr100_[0-9]*[0-9]$', embeddingType) is not None:
        index = int(re.search(r'\d+$', embeddingType).group())
        matrixGprime, radius = testcorr100(imageProductMatrix, index)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, index, r=radius, abs_tol=100)
    elif re.search(r'dblcorr_[0-9]*[0-9]$', embeddingType) is not None:
        index = int(re.search(r'\d+$', embeddingType).group())
        matrixGprime = dblcorr(imageProductMatrix, index)
        embeddingMatrix = get_embeddings_mPCA(matrixGprime, index)
    else:
        raise ValueError(embeddingType + " is not a valid embedding type")
    return embeddingMatrix


def get_eig_for_symmetric(matrixG: NDArray) -> (NDArray, NDArray):
    """
    :param matrixG: Symmetric matrix G
    :return: Eigenvalues, eigenvectors for matrix G.

    Checks that the matrix is symmetric then returns the sorted (in descending order) eigenvalues and the transposed eigenvectors
    """
    if not check_symmetric(matrixG):
        raise ValueError("Matrix G has to be a symmetric matrix")
    # The eigenvectors are already given in the form of a transposed eigenvector matrix, where the rows represent the
    # eigenvectors instead of columns
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    # Reverses the order from ascending to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors.T[::-1].T
    return eigenvalues, eigenvectors


def get_embeddings_mPCA(matrixG: NDArray, nDim=None, r=1, abs_tol=1e-5):
    """
    :param matrixG: Matrix G to be decomposed
    :param nDim: Number of dimensions of the vector embedding
    :param r: Radius of the hypersphere
    :return: An embedding matrix, with each vector having nDim dimensions. An nDim by len(MatrixG) matrix
    TLDR: Input matrix G' into the function, with the number of dimensions you want your vectors to have.
    Function will then output matrix A.

    1. Keeps the largest nDim number of eigenvalues and zeros the rest.
    2. Decomposes the matrix G into matrix A
    3. Normalize the embedding matrix

    """
    # Check and clean input
    matrixG, nDim = is_valid_matrix_g(matrixG, nDim)

    eigenvalues, eigenvectors = get_eig_for_symmetric(matrixG)

    # Checking that the matrix is positive semi-definite
    for eigenvalue in eigenvalues[:nDim]:
        if not (eigenvalue > 0 or math.isclose(eigenvalue, 0, abs_tol=abs_tol)):
            raise NonPositiveSemidefiniteError("Matrix G does not have al least nDim number of positive eigenvalues")

    # Zeros negative eigenvalues which are close to zero
    eigenvalues[0 > eigenvalues] = 0
    Droot = np.sqrt(np.diag(eigenvalues))

    # Slices the diagonal matrix to remove smaller eigenvalues
    Drootm = Droot[:nDim, :]

    matrixA = np.matmul(Drootm, eigenvectors.T)

    # Normalizing matrix A
    matrixA = normalize(matrixA, norm='l2', axis=0) * ((r + 1) ** 0.5)
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
