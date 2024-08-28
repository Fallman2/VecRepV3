import logging
import math

import cv2
import random
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.data_processing.EmbeddingFunctions import pencorr
from src.helpers.FindingEmbUsingSample import Lagrangian_Method2
import src.data_processing.ImageProducts as ip
import src.data_processing.ImageGenerators as ig
import src.visualization.Metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing.EmbeddingFunctions import get_embedding_matrix
import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as fputils
from src.data_processing.EmbeddingFunctions import pencorr, pencorr_python

"""
The purpose of this file is to test my understanding of concepts presented in the repo handed down to me from 
previous interns. I will probably use this as a test bed for things before implementing them in their own files.
In the future, I might convert this file into a worksheet of sorts to guide interns taking over.

Task 1: For all 2x2 matrices as outlined in the toy problem, find the Image Product Matrix G and then,
try to find the Embedding Matrix A.

"""

def produceG():
    img_set = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], [[0, 0], [1, 1]],
                        [[0, 1], [1, 1]], [[1, 1], [1, 1]]])
    G = []
    for mainImg in img_set:
        row = []
        for tempImg in img_set:
            row.append(ncc(mainImg, tempImg))
        G.append(row)
    return np.array(G)

def produceA(matrixG):
    eigenvalues, eigenvectors = np.linalg.eigh(matrixG)
    eigenvalues = eigenvalues[::-1]
    eigenvalues[0 > eigenvalues] = 0
    eigenvectors = eigenvectors.T[::-1].T
    Droot = np.sqrt(np.diag(eigenvalues))
    matrixA = np.matmul(Droot, eigenvectors.T)
    matrixA = normalize(matrixA, norm='l2', axis=0)
    return matrixA

"""
Task 2: Consider the following toy problem:

We wish to do similar image comparisons on triangles represented in a matrix. The triangles each fit into a 4x4 matrix
and are compared to an 8x8 main image which consists of another similar 4x4 triangle somewhere in the image (might be
in the middle).
"""
def triangleCalcs():
    smallestTriangle = [[1,0],
                        [1,1]]
    smallestTriangle = np.asarray(smallestTriangle, np.uint8)
    smallTriangle = [[0,0,0,0],
                     [0,0,0,0],
                     [1,0,0,0],
                     [1,1,0,0]]
    smallTriangle = np.asarray(smallTriangle, np.uint8)
    bigTriangle = [[1,0,0,0],
                   [1,1,0,0],
                   [1,1,1,0],
                   [1,1,1,1]]
    bigTriangle = np.asarray(bigTriangle, np.uint8)
    smallTriangleMultiplied = [[0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,1,0,0,0,0,0],
                               [0,0,1,1,0,0,0,0],
                               [0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0]]
    smallTriangleMultiplied = np.asarray(smallTriangleMultiplied, np.uint8)
    bigTriangleMultiplied = [[0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0,0],
                             [0,0,1,1,1,0,0,0],
                             [0,0,1,1,1,1,0,0],
                             [0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0]]
    bigTriangleMultiplied = np.asarray(bigTriangleMultiplied, np.uint8)
    return (ncc(bigTriangleMultiplied, smallTriangle), ncc(smallTriangleMultiplied, bigTriangle),
            ncc(smallTriangleMultiplied, smallestTriangle), ncc(bigTriangleMultiplied, smallestTriangle))

"""
Findings from task 2:
If the triangle is indeed in the middle of the matrix, then there is a finite number of triangles that we have to
compare the template images to (either way, there will be a finite number, but the finite number is much smaller
for the case of the triangle being in the middle). This number is 64, so the dimension of our embedding matrix is at
most 64. Then, we can use our usual sampling method to create matrix G or G'. However, also note the problem of finding
smaller triangles.

Some smaller triangles can be found within larger triangles, as shown above, this is further exemplified if the main
image triangles are in the middle of the image. Then, when trying to match these smaller triangles, note that the NCC
score will be 1. So, we may be able to find new constraints for the sake of the Lagrangian method.

Since more comparisons will result in an NCC score of 1, it is possible that the training set may have to be larger to 
ensure that the G created will be more reliable. 

Subsequent task:
Consider this toy problem and create code/carry out previous procedures on this set of all 4x4 triangles.

From counting, there are 48x4=192 such triangles. This has been verified using combinatorics.
Next, we implement the image set. Use the rotate function to rotate every rotationally unique triangle to get every
triangle. Then, pad the surroundings with 2 rows of 0s.
"""
def generateImageSet(mean_subtracted=False, gridwide=False):
    unpaddedImageset = []
    two_by_two = np.array([[[1, 0],[1, 1]]])
    three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
    two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
    three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [0, 0, 1]],
                               [[0, 1, 0], [1, 1, 0], [0, 0, 1]]])
    three_by_four = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0]],
                              [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
                              [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                              [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
                              [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                              [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]])
    four_by_four = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
    if not mean_subtracted:
        for tri_image in two_by_two:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(getRotationsAndPad(tri_image))
    else:
        for tri_image in two_by_two:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(getRotationsAfterMeanSubtractAndPad(tri_image))
    imageSet = []
    for tri_image in unpaddedImageset:
        imageSet.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
    if gridwide:
        for i in range(len(imageSet)):
            imageSet[i] = mean_subtract(imageSet[i])
    return np.asarray(imageSet)

def calculateTriangleG(imageSet: NDArray, imageProduct=lambda x, y: ncc(x, y)):
    G = []
    for image1 in imageSet:
        for image2 in imageSet:
            G.append(imageProduct(image1, image2))
    return np.reshape(G, (len(imageSet), len(imageSet)))

def getRotationsAndPad(tri_image: NDArray):
    ls = [padToFour(tri_image)]
    for i in range(0, 3):
        tri_image = np.rot90(tri_image)
        ls.append(padToFour(tri_image))
    return ls

def getRotationsAfterMeanSubtractAndPad(tri_image: NDArray):
    mean_subtracted = mean_subtract(padToFour(tri_image))
    ls = [mean_subtracted]
    for i in range(0, 3):
        mean_subtracted = np.rot90(mean_subtracted)
        ls.append(mean_subtracted)
    return ls

def padToFour(tri_image: NDArray):
    shape = tri_image.shape
    return np.pad(tri_image, ((4 - shape[0], 0), (0, 4 - shape[1])), constant_values=(0, 0))

def calculateTriangleAWithBF():
    G = calculateTriangleG(generateImageSet())
    G_prime = pencorr(G, 192)
    eigenvalues, eigenvectors = np.linalg.eigh(G_prime)
    eigenvalues[eigenvalues < 0] = 0
    D = np.diag(np.sqrt(eigenvalues))
    A = np.matmul(D, eigenvectors.transpose())
    return A

def calculateAfromG(G):
    G_prime = pencorr(G, 192)
    eigenvalues, eigenvectors = np.linalg.eigh(G_prime)
    eigenvalues[eigenvalues < 0] = 0
    D = np.diag(np.sqrt(eigenvalues))
    A = np.matmul(D, eigenvectors.transpose())
    return A

"""
Task: Compare triangles that you think are "similar" and see if they get a high NCC score from both
direct NCC calculation and through vector embedding.
"""

class TriangleImageSet:

    def __init__(self):
        self.fullSet = generateImageSet()
        self.two_by_two = np.array([[[1, 0],[1, 1]]])
        self.three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
        self.two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
        self.three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [0, 0, 1]],
                               [[0, 1, 0], [1, 1, 0], [0, 0, 1]]])
        self.three_by_four = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0]],
                              [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
                              [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                              [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
                              [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                              [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]])
        self.four_by_four = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
        self.G = calculateTriangleG(self.fullSet)
        self.Gprime = pencorr(self.G, 192)
        self.A = produceA(self.Gprime)

    def get2x2sample(self):
        imageSet = getRotationsAndPad(self.two_by_two[0])
        a = []
        for tri_image in imageSet:
            a.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
        return a

    def get3x2sample(self):
        list = []
        sample = random.choice(self.three_by_two)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x2sample(self):
        list = []
        sample = random.choice(self.two_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def get3x3sample(self):
        list = []
        sample = random.choice(self.three_by_three)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x4sample(self):
        list = []
        sample = random.choice(self.four_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def get4x3sample(self):
        list = []
        sample = random.choice(self.three_by_four)
        list.append(getRotationsAndPad(sample))
        return list

    def getFullSet(self):
        return self.fullSet

    def getG(self):
        return self.G

    def getGprime(self):
        return self.Gprime

    def getA(self):
        return self.A

    def generate_new_G(self, imageProduct):
        return calculateTriangleG(self.fullSet, imageProduct=imageProduct)


"""
Sanity check 1: Are rotations of images similar to each other?
    Interpretation: Not really. NCC score of rotations can go as low as 0.3.
Sanity check 2: Are larger versions of the same triangle similar to smaller versions and vice versa?
    Interpretation: Relatively, but as size differs, score differs more.
Sanity check 3: More triangle comparisons to visually similar triangles
    Interpretation: These triangles show a 0.75 or more NCC score.
Sanity check 4: Flipped triangles
    Interpretation: In the first test, the ncc score is high. However, when tested on the smaller triangle (4x3 size),
    the NCC score is only 0.5 (for both). Smaller triangles are likely more sensitive to differences.
Sanity check 5: Smaller triangles
    Interpretation: Possibly confirms previous test. All matches have NCC score around 0.6.
"""

def triangleSanityTest1():
    imageSet = TriangleImageSet()
    sample = imageSet.get2x2sample()
    firstImage = sample[0]
    b = []
    for i in sample:
        b.append(ncc(i, firstImage))
    x = get_embedding_estimate(firstImage)
    for i in sample:
        b.append((np.dot(get_embedding_estimate(i), x)))
    return firstImage, b

def triangleSanityTest2():
    two = np.pad(padToFour(np.array([[1, 0],[1, 1]])), (2, 2), constant_values=(0, 0))
    three = np.pad(padToFour(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])), (2, 2), constant_values=(0, 0))
    four = np.pad(padToFour(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])), (2, 2), constant_values=(0, 0))

    a = (ncc(two, two), ncc(two, three), ncc(two, four))
    b = (ncc(four, two), ncc(four, three), ncc(four, four))
    two2 = get_embedding_estimate(two)
    three2 = get_embedding_estimate(three)
    four2 = get_embedding_estimate(four)
    c = (np.dot(two2, two2), np.dot(two2, three2), np.dot(two2, four2))
    return a, b, c

def triangleSanityTest3():
    one = np.array([[1, 0], [1, 0], [1, 1]])
    two = np.rot90(np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]))
    three = np.rot90(np.array([[1, 1], [1, 0], [1, 0]]))
    four = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

    imageSet = [one, two, three, four]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    one2 = get_embedding_estimate(imgSet[0])
    est = []
    for tri_image in imgSet:
        est.append(np.dot(one2, get_embedding_estimate(tri_image)))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[0], imgSet[2]),
            ncc(imgSet[0], imgSet[3]), est), imgSet

def triangleSanityTest4():
    one = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    two = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
    three = np.rot90(np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]]))
    four = np.rot90(np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
    five = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    imageSet = [one, two, three, four, five]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    one2 = get_embedding_estimate(imgSet[0])
    three2 = get_embedding_estimate(imgSet[2])
    est = (np.dot(one2, get_embedding_estimate(imgSet[1])), np.dot(three2, get_embedding_estimate(imgSet[3])),
           np.dot(three2, get_embedding_estimate(imgSet[4])))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[2], imgSet[2]),
            ncc(imgSet[2], imgSet[3]), ncc(imgSet[2], imgSet[4]), est), imgSet

def triangleSanityTest5():
    one = np.array([[1, 0], [1, 0], [0, 1]])
    two = np.array([[0, 1], [1, 0], [1, 0]])
    three = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    four = np.rot90(np.array([[0, 0, 1, 1], [1, 0, 0, 0]]))
    five = np.rot90(two, k=-1)

    imageSet = [one, two, three, four, five]
    imgSet = []
    for tri_image in imageSet:
        imgSet.append(np.pad(padToFour(tri_image), (2, 2), constant_values=(0, 0)))
    est = []
    one2 = get_embedding_estimate(imgSet[0])
    for tri_image in imgSet:
        est.append(np.dot(one2, get_embedding_estimate(tri_image)))
    return (ncc(imgSet[0], imgSet[0]), ncc(imgSet[0], imgSet[1]), ncc(imgSet[0], imgSet[2]),
            ncc(imgSet[0], imgSet[3]), ncc(imgSet[0], imgSet[4]), est), imgSet

"""
Mean subtracted tests: Mean subtracted triangles give a generally quite similar Relative Positioning Score as non
mean subtracted triangles. At k=3, normal triangles win out, but at k=5, mean subtracted triangles give a slightly
higher score.
Next, try mean subtracting across the entire grid.

When mean subtracting across the whole grid, Relative Positioning score is strictly worse for both k=3 and k=5
"""

# This estimate is using the Lagrangian method and is the same as the one in SampleEstimator.py
def get_embedding_estimate(image):
    triangleImageSet = TriangleImageSet()
    trainingImageSet = triangleImageSet.getFullSet()
    imageProductVector = ip.calculate_image_product_vector(image, trainingImageSet, ncc)
    estimateVector = Lagrangian_Method2(triangleImageSet.getA(), imageProductVector)[0]
    return estimateVector

# This NCC calculation is the same as the one in ImageProducts.py
def ncc(mainImg: NDArray, tempImg: NDArray):

    if np.count_nonzero(mainImg) == 0:
        if np.count_nonzero(tempImg) == 0:
            return 1
        return 0

    mainImg = np.pad(mainImg, max(len(mainImg), len(mainImg[0])), 'wrap')

    mainImg = np.asarray(mainImg, np.single)
    tempImg = np.asarray(tempImg, np.single)

    corr = cv2.matchTemplate(mainImg, tempImg, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

    return max_val

def mean_subtract(matrix: NDArray):
    mean = np.mean(matrix)
    return matrix - np.ones(matrix.shape) * mean

def generate_diagnostics(image_type, image_product, k=5, embedding=None, weight=None, filters=None):
    """
    :param image_type: Image Set to investigate
    :param image_product: Image Product to investigate
    :return: Returns several metric which we may want to observe. To be updated over time.
    :returns G: The Image Product Matrix
    :returns G_prime: The Nearest Correlation Matrix found using Pencorr
    :returns A: The embedding matrix found by decomposition
    :returns x: An array of tuples containing the minimum and maximum values found in corresponding rows of A
    :returns r: The range of values from x
    :returns s: The sum of all ranges. Rough estimator of area of hypersphere occupied by embeddings
    :returns prod: The product of all elements of r. Another rough estimator of proportion of Hypersphere occupied
    :returns nonzero: Number of non-zero elements of r. Indicates the number of dimensions the embeddings occupy
    :returns eigenvalues/eigenvectors: Eigenvalues and eigenvectors of G_prime
    """

    if filters is None:
        filters = []
    if weight is None:
        weight = ""
    image_set = utils.generate_filtered_image_set(image_type, filters, fputils.get_image_set_filepath(image_type, filters))
    G = utils.generate_image_product_matrix(image_set, image_product, fputils.get_image_product_filepath(image_type, filters, image_product))
    if embedding is None:
        embedding = "pencorr_" + str(len(image_set))
    weightingFilepath = fputils.get_weighting_matrix_filepath(image_type, filters, weight, image_product)
    weightMatrix = utils.generate_weighting_matrix(G, image_set, weight, weightingFilepath, fputils.get_image_product_filepath(image_type, filters, image_product))
    A = utils.generate_embedding_matrix(G, embedding, fputils.get_embedding_matrix_filepath(image_type, filters, image_product, embedding), weight=weightMatrix)
    G_prime = np.matmul(np.atleast_2d(A).T, np.atleast_2d(A))
    x = np.array([(np.min(b), np.max(b)) for b in A])   # Range of values which the embeddings take in each dimension
    r = np.array([np.max(b) - np.min(b) for b in A])    # Magnitude of this range
    s = sum([np.max(b) - np.min(b) for b in A])         # Sum of all ranges, to gauge how much of the Hypersphere we are using
    nonzero = np.count_nonzero(r)                       # Number of dimensions used
    prod = np.sum([math.log10(i) for i in r[r != 0]])
    # prod2 = prod - math.log10((math.pi ** (nonzero / 2)) / math.gamma(1 + nonzero / 2))
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    above_one = len(eigenvalues[eigenvalues > 1])
    post_eigenvalues, post_eigenvectors = np.linalg.eigh(G_prime)
    k_score = metrics.get_mean_normed_k_neighbour_score(G, G_prime, k)
    vector_range = np.array([(np.min(b), np.max(b)) for b in eigenvectors])
    return G, G_prime, A, x, r, s, prod, nonzero, eigenvalues, eigenvectors, k_score, embedding, post_eigenvalues, post_eigenvectors, above_one, vector_range

def compare_diagnostics(image_set, image_product_list, embeddings, weights=None, k=5, filters=None):
    if filters is None:
        filters = []
    if weights is None:
        weights = [""]

    data = {}
    for image_product in image_product_list:
        data[image_product] = {}
        for embedding in embeddings:
            data[image_product][embedding] = {}
            for weight in weights:
                data[image_product][embedding][weight] = {}
                G, G_prime, A, x, ranges, s, prod, nonzero, eigenvalues, eigenvectors, k_score, emb, peigenvalues, peigenvectors, above_one, vector_range = generate_diagnostics(image_set, image_product, weight=weight, k=k, filters=filters, embedding=embedding)
                data[image_product][embedding][weight]["image_product"] = image_product
                data[image_product][embedding][weight]["A"] = A
                data[image_product][embedding][weight]["G"] = G
                data[image_product][embedding][weight]["Gprime"] = G_prime
                data[image_product][embedding][weight]["embedding"] = emb
                data[image_product][embedding][weight]["sum"] = s
                data[image_product][embedding][weight]["non_zero"] = nonzero
                data[image_product][embedding][weight]["non_negative"] = np.sum([eigenvalues >= 0])
                data[image_product][embedding][weight]["prod"] = prod
                data[image_product][embedding][weight]["k_scores"] = k_score
                data[image_product][embedding][weight]["rng"] = ranges
                data[image_product][embedding][weight]["eigenvectors"] = eigenvectors
                data[image_product][embedding][weight]["eigenvalues"] = eigenvalues
                data[image_product][embedding][weight]["peigenvectors"] = peigenvectors
                data[image_product][embedding][weight]["peigenvalues"] = peigenvalues
                data[image_product][embedding][weight]["above_one"] = above_one
    return data

def display_df(data, image_product_list, display, embeddings, weights):
    displayed_data = {"Image Product": [], "Embedding": [], "Weight": []}
    for i in display:
        displayed_data[i] = []
        for j in image_product_list:
            for k in embeddings:
                for l in weights:
                    displayed_data[i].append(data[j][k][l][i])
                    displayed_data["Image Product"].append(j)
                    displayed_data["Embedding"].append(k)
                    displayed_data["Weight"].append(l)
    return pd.DataFrame(displayed_data)

def display_eigenvalues(data, image_product_list, embeddings, weights, p=0):
    stats = {"large": {}, "sum": {}}
    if p < 2:
        category = "eigenvalues"
        if p:
            category = "p" + category
        for image_product in image_product_list:
            for embedding in embeddings:
                stats["large"][embedding] = {}
                stats["sum"][embedding] = {}
                for weight in weights:
                    eigenvalues = data[image_product][embedding][weight][category]
                    s = sum(eigenvalues)
                    sorted(eigenvalues)
                    large = np.array(eigenvalues)[eigenvalues > 100].tolist()
                    """
                    if len(large) > 0:
                        eigenvalues = eigenvalues[:-len(large)]
                    """
                    label = image_product + "_" + weight
                    plt.plot(eigenvalues, label=label)
                    stats["large"][embedding][image_product + "_" + weight] = large
                    stats["sum"][embedding][image_product + "_" + weight] = s
    else:
        categories = ["eigenvalues", "peigenvalues"]
        for image_product in image_product_list:
            for embedding in embeddings:
                stats["large"][embedding] = {}
                stats["sum"][embedding] = {}
                for weight in weights:
                    for category in categories:
                        stats["large"][embedding][image_product + "_" + weight] = {}
                        stats["sum"][embedding][image_product + "_" + weight] = {}
                        eigenvalues = data[image_product][weight][category]
                        s = sum(eigenvalues)
                        sorted(eigenvalues)
                        large = np.array(eigenvalues)[eigenvalues > 100].tolist()
                        if len(large) > 0:
                            eigenvalues = eigenvalues[:-len(large)]
                        label = image_product + "_" + weight + "_" + category
                        plt.plot(eigenvalues, label=label)
                        stats["large"][embedding][image_product + "_" + weight][category] = large
                        stats["sum"][embedding][image_product + "_" + weight][category] = s
    plt.legend(loc="lower right")
    plt.show()
    return stats




def display_multiple(data, image_product_list, display, embeddings):
    displayed_data = {"image_product": [], "embedding": []}
    for j in image_product_list:
        for k in embeddings:
            for i in display:
                if i not in displayed_data:
                    displayed_data[i] = []
                displayed_data[i].append(data[j][k][i])
                displayed_data["image_product"].append(j)
                displayed_data["embedding"].append(k)
    return pd.DataFrame(data)


def compare_image_products(image_set, image_product_list, embeddings=None):
    if embeddings is None:
        embeddings = [("pencorr_" + str(len(image_set)))]
    data = {
        "image_product": [],
        "embedding": [],
        "sum": [],
        "non_zero": [],
        "non_negative": [],
        "prod": [],
        "k_scores": []
    }
    for image_product in image_product_list:
        img_prod = ip.get_image_product(image_product)
        for embedding in embeddings:
            (G, G_prime, A, x, r, s, prod, nonzero, eigenvalues, eigenvectors, k_score, emb,
             peighenvalues, peigenvectors, above_one, vector_range) = generate_diagnostics(image_set, image_product, embedding=embedding)
            data["image_product"].append(image_product)
            data["embedding"].append(emb)
            data["sum"].append(s)
            data["non_zero"].append(nonzero)
            data["non_negative"].append(np.sum([eigenvalues >= 0]))
            data["prod"].append(prod)
            data["k_scores"].append(k_score)
    df = pd.DataFrame(data)
    return df

def plot_k_on_values(k: int, image_type: str, image_product_list: list, plot=None, embeddings=None):
    if embeddings is None:
        embeddings = [("pencorr_" + str(len(ig.get_image_set(image_type))))]
    data = {
        "image_product": [],
        "embedding": [],
        "sum": [],
        "non_zero": [],
        "non_negative": [],
        "prod": [],
        "k_scores": []
    }
    for image_product in image_product_list:

        for embedding in embeddings:
            (G, G_prime, A, x, r, s, prod, nonzero, eigenvalues, eigenvectors, k_score, red,
            peighenvalues, peigenvectors, above_one, vector_range) = generate_diagnostics(image_type, image_product, k=k, embedding=embedding)

            data["image_product"].append(image_product)
            data["embedding"].append(embedding)
            data["sum"].append(s)
            data["non_zero"].append(nonzero)
            data["non_negative"].append(np.sum([eigenvalues >= 0]))
            data["prod"].append(prod)
            data["k_scores"].append(k_score)
    if plot is None:
        plot = "non_zero"
    plt.plot(data[plot], data["k_scores"], "r+")
    plt.plot(np.unique(data[plot]), np.poly1d(np.polyfit(data[plot], data["k_scores"], 1))(np.unique(data[plot])))
    idealPlot = [1 for i in range(len(data[plot]))]  # for plotting the max possible score
    plt.plot(data[plot], idealPlot, color='b', linestyle=':', label="Ideal")
    plt.show()
    df = pd.DataFrame(data)
    return df


def k_means_clustering(image_type: str, image_product: str, embedding: str, weight: str, filters, k=3):
    image_set = utils.generate_filtered_image_set(image_type, filters, fputils.get_image_set_filepath(image_type, filters))
    G = utils.generate_image_product_matrix(image_set, image_product, fputils.get_image_product_filepath(image_type, filters, image_product))
    weightingFilepath = fputils.get_weighting_matrix_filepath(image_type, filters, weight, image_product)
    weightMatrix = utils.generate_weighting_matrix(G, image_set, weight, weightingFilepath, fputils.get_image_product_filepath(image_type, filters, image_product))
    A = utils.generate_embedding_matrix(G, embedding, fputils.get_embedding_matrix_filepath(image_type, filters, image_product, embedding), weight=weightMatrix)
    kmeans = KMeans(n_clusters=k).fit(A.T)
    return A, kmeans, lambda x: estimateEmbedding(x, image_set, ip.get_image_product(image_product), A)


def k_means_cluster_prediction(testing_image_set: str, filters, kMeans, estimator):
    testing_set = utils.generate_filtered_image_set(testing_image_set, filters, fputils.get_image_set_filepath(testing_image_set, filters))
    embeddings = []
    for i in testing_set:
        embeddings.append(estimator(i))
    embeddings = np.array(embeddings, dtype="double")
    classification = kMeans.predict(embeddings)
    return testing_set, classification


def estimateEmbedding(imageInput, ImageSet, imageProduct, embeddingMatrix) -> NDArray:
    """
    :param imageInput: Takes in an image with the same dimensions as images in the image sample
    :return: A vector embedding of the input image generated using the image sample. Method used is by minimizing
    the error between the dot product results and the image product vector.
    """
    imageProductVector = ip.calculate_image_product_vector(imageInput, ImageSet, imageProduct)
    estimateVector = Lagrangian_Method2(embeddingMatrix, imageProductVector)[0]
    return estimateVector


def find_plateau_rank(image_sets: list, filters: list, image_product_list, embeddings, weights, k=5, prox=3):
    data = {"Image Set": image_sets, "Image Products": image_product_list, "Embeddings": embeddings, "Weights": weights,
            "K_scores": [], "Set Size": [], "Plateau Rank": [], "Non_zero": []}
    for index, image_type in enumerate(image_sets):
        logging.info("Investigating " + image_type)
        image_product = image_product_list[index]
        weight = weights[index]
        embedding = embeddings[index]
        image_set = utils.generate_filtered_image_set(image_type, filters,
                                                      fputils.get_image_set_filepath(image_type, filters))
        data["Set Size"].append(len(image_set))
        high = len(image_set)
        low = 0
        selected_rank = high
        max_k_score = 2
        iterations = 0
        same_rank = 0
        score_change = False
        max_score_rank = []
        while high - low > prox:
            logging.info("Starting iteration " + str(iterations + 1))
            G = utils.generate_image_product_matrix(image_set, image_product,
                                                    fputils.get_image_product_filepath(image_type, filters, image_product))
            selected_embedding = embedding + "_" + str(selected_rank)
            weightingFilepath = fputils.get_weighting_matrix_filepath(image_type, filters, weight, image_product)
            weightMatrix = utils.generate_weighting_matrix(G, image_set, weight, weightingFilepath,
                                                           fputils.get_image_product_filepath(image_type, filters,
                                                                                              image_product))
            A = utils.generate_embedding_matrix(G, selected_embedding,
                                                fputils.get_embedding_matrix_filepath(image_type, filters, image_product,
                                                                                      selected_embedding), weight=weightMatrix)
            G_prime = np.matmul(np.atleast_2d(A).T, np.atleast_2d(A))
            k_score = metrics.get_mean_normed_k_neighbour_score(G, G_prime, k)
            if not score_change:
                if iterations == 0:
                    max_k_score = k_score
                    data["K_scores"].append(max_k_score)
                    nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in A]))
                    data["Non_zero"].append(nonzero)
                    high = nonzero
                    low = nonzero // 2
                elif k_score == max_k_score:
                    high = low
                    low = high // 2
                else:
                    score_change = True
                    low = ((high - low) // 2) + low
            elif k_score != max_k_score:
                low = ((high - low) // 2) + low
                max_score_rank = []
                same_rank = 0
            elif same_rank == 2:
                high = max_score_rank[0]
                iterations += 1
                logging.info("Finishing iteration" + str(iterations))

                break
            else:
                max_score_rank.append(low)
                diff = (high - low) // 4
                low += diff
                same_rank += 1


            selected_rank = low
            iterations += 1
            logging.info("Finishing iteration" + str(iterations))
            logging.info("Next Rank " + str(low))
        logging.info("Plateau rank " + str(high))
        data["Plateau Rank"].append(high)
    return pd.DataFrame(data)


def display_plateau_rank(data):
    plt.plot(data["Set Size"], data["Plateau Rank"])
    plt.show()

def test_pencorr():
    image_type = "2bin"
    filters = ["unique"]
    image_product = "ncc"
    weight = ""
    image_set = utils.generate_filtered_image_set(image_type, filters,
                                                  fputils.get_image_set_filepath(image_type, filters))
    G = utils.generate_image_product_matrix(image_set, image_product,
                                            fputils.get_image_product_filepath(image_type, filters, image_product))
    Gprime1 = pencorr(G, 7)
    Gprime2 = pencorr_python(G, 7)
    """
    A1 = utils.generate_embedding_matrix(G, "pencorr_7",
                                        fputils.get_embedding_matrix_filepath(image_type, filters, image_product,
                                                                              "pencorr_7"), weight=None)
    A2 = utils.generate_embedding_matrix(G, "pencorr_python_7",
                                        fputils.get_embedding_matrix_filepath(image_type, filters, image_product,
                                                                              "pencorr_python_7"), weight=None)
    """
    return Gprime1, Gprime2, np.max(np.max(np.abs(Gprime1 - Gprime2))), G


def show_graph(data):
    fig = plt.figure()
    fig.suptitle("k-scores on Image Set Sizes ")
    axes = plt.axes()
    set_sizes = np.array(data["Set Size"])
    k_scores = np.array(data["K_scores"])
    indexes = np.argsort(set_sizes)
    set_sizes = set_sizes[indexes]
    k_scores = k_scores[indexes]
    axes.plot(set_sizes, k_scores)
    axes.set_xlabel("Image Set Sizes")
    axes.set_ylabel("k-score")
    plt.show()


"""
def generate_weightings(matrixG: NDArray, index, k=5, base=None, filters=None) -> NDArray:
    
    :param matrixG: Matrix G to be decomposed
    :return: Weightings through which to run weighted pencorr
    
    if filters is None:
        filters = []
    if base is not None:
        filepath = fpUtils.get_image_product_filepath(base, filters, "ncc")
        imageSet = utils.generate_filtered_image_set(base, filters, fpUtils.get_image_set_filepath(base, filters))
        matrixG = utils.generate_image_product_matrix(imageSet, "ncc", filepath,
                                                      False)
        return matrixG ** index

    if index == 0:
        return np.ones((len(matrixG), len(matrixG)))
    elif index == 1:
        return matrixG
    elif index == 2:
        return matrixG ** 2
    elif index == 3:
        return matrixG ** 3
    elif index == 4:
        nbr_arr = matrixG.transpose()
        k = k + 1
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = nbr_arr[row][n]
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 5:
        nbr_arr = matrixG.transpose()
        k = k + 5
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = 1
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 6:
        nbr_arr = matrixG.transpose()
        k = k + 5
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = nbr_arr[row][n]
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 7:
        return matrixG ** 5
    elif index == 8:
        filepath = fpUtils.get_image_product_filepath("triangle", [], "ncc")
        matrixG = utils.generate_image_product_matrix(get_triangle_image_set(), "ncc", filepath,
                                                      False)
        return matrixG ** 20
    elif index == 10:
        nbr_arr = matrixG.transpose()
        k = k + 1
        weight_arr = np.zeros_like(matrixG)
        for row in range(len(nbr_arr)):
            max_index = np.argpartition(matrixG[row], -k)[-k:]
            kth_largest = nbr_arr[row][max_index[0]]
            kth_element = np.where(nbr_arr[row] == kth_largest)
            max_index = np.union1d(max_index, kth_element)
            for n in max_index:
                weight_arr[row][n] = 1
        weight_arr = np.asarray(weight_arr)
        return weight_arr.transpose()
    elif index == 11:
        return 10 ** (matrixG - 1)
    elif index == 12:
        return 20 ** (matrixG - 1)
    elif index == 13:
        return 30 ** (matrixG - 1)
    elif index == 20:
        filepath = fpUtils.get_image_product_filepath("triangle", [], "ncc")
        matrixG = utils.generate_image_product_matrix(get_triangle_image_set(), "ncc", filepath,
                                                      False)
        return matrixG
    else:
        raise ValueError(str(index) + "is not a valid weighting index")
"""
"""
Some useful links and ideas
https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity
https://learn.microsoft.com/en-us/azure/search/vector-search-overview#approximate-nearest-neighbors

Vector similarity can help to determine close images or similar images. Cosine similarity is equal to NCC score.
Consider any image, then calculate the closest image?

For similarity metric, consider developing relative positioning score with more specific metric calculations.

Properties of embeddings:
Note that xtx = 1. So, arranging them into a diagonal matrix, we get Dx squared = ??
"""

"""
def get_triangle_image_set(mean_subtracted=False, gridwide=False):
    unpaddedImageset = []
    two_by_two = np.array([[[1, 0], [1, 1]]])
    three_by_two = np.array([[[1, 0], [1, 0], [1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1], [1, 0], [1, 0]],
                             [[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0], [1, 0]]])
    two_by_four = np.array([[[1, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 0, 0]], [[1, 1, 1, 1], [0, 0, 1, 0]],
                            [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 0, 1, 1], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 1]],
                            [[0, 1, 1, 1], [1, 0, 0, 0]], [[1, 1, 1, 0], [0, 0, 0, 1]]])
    three_by_three = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
                               [[1, 0, 0], [1, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 1, 0], [1, 0, 0]],
                               [[0, 1, 0], [1, 1, 0], [0, 0, 1]]])
    three_by_four = np.array([[[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0]],
                              [[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]], [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
                              [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                              [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
                              [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                              [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                              [[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              [[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]])
    four_by_four = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             [[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
                             [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                             [[0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])
    if not mean_subtracted:
        for tri_image in two_by_two:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(get_rotations_and_pad(tri_image))
    else:
        for tri_image in two_by_two:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_two:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in two_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_three:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in three_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
        for tri_image in four_by_four:
            unpaddedImageset.extend(get_rotations_of_mean_subtracted_and_padded(tri_image))
    imageSet = []
    for tri_image in unpaddedImageset:
        imageSet.append(np.pad(tri_image, (2, 2), constant_values=(0, 0)))
    if gridwide:
        for i in range(len(imageSet)):
            imageSet[i] = imageSet[i] - np.ones(imageSet[i].shape) * np.mean(imageSet[i])
    return np.asarray(imageSet)


def get_rotations_and_pad(tri_image: NDArray):
    ls = [pad_to_four(tri_image)]
    for i in range(0, 3):
        tri_image = np.rot90(tri_image)
        ls.append(pad_to_four(tri_image))
    return ls


def pad_to_four(tri_image: NDArray):
    shape = tri_image.shape
    return np.pad(tri_image, ((4 - shape[0], 0), (0, 4 - shape[1])), constant_values=(0, 0))


def get_rotations_of_mean_subtracted_and_padded(tri_image: NDArray):
    tri_image = pad_to_four(tri_image)
    mean_subtracted = tri_image - np.ones(tri_image.shape) * np.mean(tri_image)
    ls = [mean_subtracted]
    for i in range(0, 3):
        mean_subtracted = np.rot90(mean_subtracted)
        ls.append(mean_subtracted)
    return ls
    
    def straight_lines():
    vertical_lines = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
    horizontal_lines = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    diagonal_lines = [[0, 5, 10, 15], [3, 6, 9, 12]]
    straights = []
    straights.extend(vertical_lines)
    straights.extend(horizontal_lines)
    straights.extend(diagonal_lines)
    return straights
    
def fill_shape(combination: list, size):
    matrix = np.zeros((size, size), dtype=int)
    indexes = list(range(0, size ** 2))
    for index in indexes:
        if index in combination:
            continue
        position = (index // 4, index % 4)
        top_left = False
        top_right = False
        bottom_left = False
        bottom_right = False
        left = False
        right = False
        up = False
        down = False
        for selected in combination:
            # Check if surrounded on diagonals
            selected_position = (selected // 4, selected % 4)
            top_left = selected_position[0] < position[0] and selected_position[1] < position[1]
            top_right = selected_position[0] < position[0] and selected_position[1] > position[1]
            bottom_left = selected_position[0] > position[0] and selected_position[1] < position[1]
            bottom_right = selected_position[0] > position[0] and selected_position[1] > position[1]
            diamond = top_left and top_right and bottom_left and bottom_right




def fill_shape2(matrix: NDArray):
    change = True
    while change:
        before = np.copy(matrix)
        for x in range(4):
            for y in range(4):
                if matrix[x][y]:
                    continue
                x_m1 = matrix[max(0, x - 1)][y]
                x_m2 = matrix[max(0, x - 2)][y]
                x_p1 = matrix[min(x + 1, 3)][y]
                x_p2 = matrix[min(x + 2, 3)][y]
                y_m1 = matrix[x][max(0, y - 1)]
                y_m2 = matrix[x][max(0, y - 2)]
                y_p1 = matrix[x][min(y + 1, 3)]
                y_p2 = matrix[x][min(y + 2, 3)]

                # Horizontal check
                left = y_m1 or y_m2
                right = y_p1 or y_p2
                if left and right:
                    matrix[x][y] = 1
                    continue

                # Vertical check
                up = x_m1 or x_m2
                down = x_p1 or x_p2
                if up and down:
                    matrix[x][y] = 1
                    continue

                x_d1 = x - 1
                x_d2 = x - 2
                y_d1 = y - 1
                y_d2 = y - 2
                x_d3 = x + 1
                y_d3 = y + 1
                x_d4 = x + 2
                y_d4 = y + 2
                if x_d1 < 0 or y_d1 < 0:
                    x_d1 = x
                    y_d1 = y
                if x_d2 < 0 or y_d2 < 0:
                    x_d2 = x_d1
                    y_d2 = y_d1
                if x_d3 > 3 or y_d3 > 3:
                    x_d3 = x
                    y_d3 = y
                if x_d4 > 3 or y_d4 > 3:
                    x_d4 = x_d3
                    y_d4 = y_d3
                # Diagonal check for top left to bottom right
                diag1 = matrix[x_d1][y_d1]
                diag2 = matrix[x_d2][y_d2]
                diag3 = matrix[x_d3][y_d3]
                diag4 = matrix[x_d4][y_d4]
                upleft = diag1 or diag2
                downright = diag3 or diag4
                if upleft and downright:
                    matrix[x][y] = 1
                    continue

                x_d1 = x - 1
                x_d2 = x - 2
                y_d1 = y - 1
                y_d2 = y - 2
                x_d3 = x + 1
                y_d3 = y + 1
                x_d4 = x + 2
                y_d4 = y + 2
                if x_d1 < 0 or y_d3 > 3:
                    x_d1 = x
                    y_d3 = y
                if x_d2 < 0 or y_d4 > 3:
                    x_d2 = x_d1
                    y_d4 = y_d3
                if x_d3 > 3 or y_d1 < 0:
                    x_d3 = x
                    y_d1 = y
                if x_d4 > 3 or y_d2 < 0:
                    x_d4 = x_d3
                    y_d2 = y_d1
                diag5 = matrix[x_d3][y_d1]
                diag6 = matrix[x_d4][y_d2]
                diag7 = matrix[x_d1][y_d3]
                diag8 = matrix[x_d2][y_d4]
                downleft = diag5 or diag6
                upright = diag7 or diag8
                # Diagonal check for bottom left to top right
                if downleft and upright:
                    matrix[x][y] = 1
        if np.array_equal(before, matrix):
            change = False
    return matrix

def remove_translationally_similar(imageSet: NDArray, comparisonSet: NDArray) -> NDArray:
    translations = set()
    squareLength = len(imageSet[0])
    unique = []
    for matrix in comparisonSet:
        # All translational invariant permutations for given nxn matrix
        for dr in range(squareLength):
            matrix = np.roll(matrix, 1, axis=0)  # shift 1 place in horizontal axis
            for dc in range(squareLength):
                matrix = np.roll(matrix, 1, axis=1)  # shift 1 place in vertical axis
                to_store = np.reshape(matrix, (1, squareLength ** 2))
                translations.add(tuple(to_store[0]))  # store in dictionary
    for image in imageSet:
        original_image = np.copy(image)
        original_image = np.reshape(original_image, (1, squareLength ** 2))
        if tuple(original_image[0]) not in translations:
            unique.append(image)
    unique = np.array(unique)
    return unique


P = [[0, 1, 0, 0, 0, 0, 0], [0.3859448175887949, 0, 0.7881884775407473, 4.062272245138956e-16, 8.626410668112123e-16, 0.2411447870341981, -0.4143123378980911],[0.3940133629363345, 0, 0.05234146489679808, -0.5052294202507127, 0.6414124254953245, -0.1054020758257833, 0.4052623553021015], [0.3940133629363345, 0, 0.05234146489679797, 0.8080941649073011, 0.1168352999287392, -0.1054020758257833, 0.4052623553021015], [0.3940133629363345, 0, 0.05234146489679953, -0.3028647446565881, -0.7582477254240636, -0.1054020758257835, 0.4052623553021014], [0.4572075477640435, 0, -0.3323544760478447, -1.908357133166104e-16, -2.360750301038771e-16, -0.6063715907269462, -0.5592988954734542], [0.4198431439558621, 0, -0.5099809205061826, -3.300708646605335e-16, -6.871755861442245e-16, 0.7354140961936693, -0.1510539715220102]]

Omega12 = np.array([0.9748609588990274, 0.8982324509985508, 0.8351324522480089, 0.8152649185396945, 0.8152649185396944, 0.5628103111570243])
I = np.array(list(range(7)))
J = I
J = I.copy()
P = [[0, 1, 0, 0, 0, 0, 0], [0.3859448175887949, 0, 0.7881884775407473, 4.062272245138956e-16, 8.626410668112123e-16, 0.2411447870341981, -0.4143123378980911],]
P = [[0, 1, 0, 0, 0, 0, 0], [0.3859448175887949, 0, 0.7881884775407473, 4.062272245138956e-16, 8.626410668112123e-16, 0.2411447870341981, -0.4143123378980911],[0.3940133629363345, 0, 0.05234146489679808, -0.5052294202507127, 0.6414124254953245, -0.1054020758257833, 0.4052623553021015], [0.3940133629363345, 0, 0.05234146489679797, 0.8080941649073011, 0.1168352999287392, -0.1054020758257833, 0.4052623553021015], [0.3940133629363345, 0, 0.05234146489679953, -0.3028647446565881, -0.7582477254240636, -0.1054020758257835, 0.4052623553021014], [0.4572075477640435, 0, -0.3323544760478447, -1.908357133166104e-16, -2.360750301038771e-16, -0.6063715907269462, -0.5592988954734542], [0.4198431439558621, 0, -0.5099809205061826, -3.300708646605335e-16, -6.871755861442245e-16, 0.7354140961936693, -0.1510539715220102]]
P = np.array(P)
from src.matlab_functions.CorMat3Mex import precond_matrix
c = precond_matrix(I, J, Omega12, P)

Omega12 = np.reshape(Omega12, (6, 1))
c = precond_matrix(I, J, Omega12, P)
b = np.reshape(np.array([0, -0.01944806094590201, -0.0186077174215693, -0.0186077174215693,-0.0186077174215693, -0.035441206452832, -0.002585144779165161]), (7, 1))
tolCG = 0.0100000
maxitCG = 200
from src.matlab_functions.CorMat3Mex import pre_cg
d,flag,relres,iterk = pre_cg(b, I, J, tolCG, maxitCG, Omega12, P, c)
b = np.reshape(b, (7,))
d,flag,relres,iterk = pre_cg(b, I, J, tolCG, maxitCG, Omega12, P, c)

lambda_ = np.array([4.393539589271419, 1, 0.5739059876660206, 0.49999994, 0.4999999399999998, 0.1458521075051682, -0.1132975644426071])
from src.matlab_functions.CorMat3Mex import omega_mat
omg2 = omega_mat(lambda_)
b0 = np.array([1, 1, 1, 1, 1, 1, 1])

y = np.array([0, -0.02333045063092509, -0.02222178389745793, -0.02222178389745791, -0.02222178389745792, -0.04999746387883026, -0.002908917137146005])
lambda2_ = np.array([4.368802085317165, 1, 0.553359312640923, 0.4777781561025424, 0.477778156102542, 0.1238580870106187, -0.1444779805130633])
P2 = [[0, 1, 0, 0, 0, 0, 0], [0.3860979180914749, 0, 0.7823566210273568, 1.143677652154381e-15, 4.507432141027466e-16, -0.2811590368467042, -0.3997450577299473], [0.3942110912036332, 0, 0.05385380090694935, 0.6096874602327108, -0.5430910305847937, 0.1224038341150691, 0.400060018531194], [0.3942110912036332, 0, 0.05385380090694925, 0.7995503442227391, 0.1224038341150691, 0.400060018531194], [0.3942110912036332, 0,0.05385380090695137, -0.7751743591702585, -0.2564593136379455, 0.1224038341150694, 0.4000600185311938], [0.4545498392178004, 0, -0.3113965607444985, -1.789013409046881e-16, 5.164312464437969e-17, 0.5928075879905838, -0.5873634217793575], [0.4220256959340548, 0 , -0.5312716855571905, -1.016222477805392e-15, -3.726576937748324e-16, -0.7242794100359415, -0.1227356683877462]]
P2[3].insert(3, 0.1654868989375476)
P2

P2 = np.array(P2)
X = [[], [], [], [], [], [], []]
X[0].extend([1, 0, 0, 0, 0, 0, 0])
X[1].extend([0, 0.9766695493690749, 0.70710677, 0.70710677, 0.70710677, 0.57735026, 0.5])
X[2].extend([0, 0.70710677, 0.9777782161025421, 0.50000006, 0.50000006, 0.81649661, 0.70710677])
X[3].extend([0, 0.70710677, 0.50000006, 0.9777782161025421, 0.50000006, 0.81649661, 0.70710677])
X[4].extend([0, 0.70710677, 0.50000006, 0.50000006, 0.9777782161025421, 0.81649661, 0.70710677])
X[5].extend([0, 0.57735026, 0.81649661, 0.81649661, 0.81649661, 0.9500025361211697, 0.86602545])
X[6].extend([0, 0.5, 0.70710677, 0.70710677, 0.70710677, 0.86602545, 0.997091082862854])
X = np.array(X)
from src.matlab_functions.CorMat3Mex import gradient
f, Fy = gradient(y, I, J, lambda2_, P, X, b0)

"""