import logging
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

import src.data_processing.Utilities as utils
import src.helpers.FilepathUtils as fputils
import src.visualization.Metrics as metrics
from src.data_processing.SampleEstimator import SampleEstimator
from src.data_processing.SampleTester import SampleTester
from visualization import GraphEstimates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def investigate_tester_rank_constraint(*, imageSet: NDArray, imageProductType: str, sampleSize: int, testSize: int,
                                       testPrefix: str, startingConstr: int, endingConstr: int, increment=1,
                                       specifiedKArr=None, plotFrob=True, weight=""):
    """
    :param specifiedKArr: value of k for the k neighbour score
    :param imageSet: Set of images used to the test and sample image sets. Currently, the training set takes from the front
    of the image set, and the test set takes from the tail of the image set TODO Do a proper monte carlo simulation.
    :param imageProductType: Image product type to investigate
    :param startingConstr: Starting lowest rank constraint to start the sweep inclusive
    :param endingConstr: Final largest rank constraint to end the sweep inclusive
    :param increment: Increment in the rank constraint sweep
    :param testPrefix: Used as a prefix to all the test names
    :param testSize: Size of the test set
    :param sampleSize: Size of the sample set
    :param plotFrob: If True, also plots frob error against rank
    :return: Uses the penncorr method to generate embeddings for different rank constraints
    Makes a graph of the average neighbour score against rank_constraint and
    average frobenius distance against rank_constraint
    Remember to use plt.show() to display plots

    Aims to answer the question: How does the rank constraint affect the error of the embeddings generated by penncorr?
    """
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]

    aveFrobDistanceArr = []
    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[] for i in specifiedKArr]
    rankConstraints = list(range(startingConstr, endingConstr + 1, increment))

    # For each rank in the sweep, generate a SampleTester and add its results to the array
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + " of " + str(endingConstr))
        embType = "pencorr_" + str(rank)
        sampleName = testPrefix + "_sample_" + str(rank) + " of " + str(endingConstr)
        testName = testPrefix + "_test_" + str(rank) + " of " + str(endingConstr)

        aveNeighArr = [[] for i in specifiedKArr]
        frobDistanceArr = []
        # TODO Start Random Sampling Here
        for j in range(5):
            # Taking training and testing samples as random samples of the image set
            testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSize)

            # Generating a sampleEstimator and SampleTester with the input parameters
            sampleEstimator = SampleEstimator(sampleName=sampleName + "_" + str(j), trainingImageSet=trainingSample, embeddingType=embType,
                                          imageProductType=imageProductType, weight=weight)
            sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)


            # For each k to be investigated, append the respective k neighbour score
            for i in range(len(specifiedKArr)):
                k = specifiedKArr[i]
                aveNeighArr[i].append(metrics.get_mean_normed_k_neighbour_score(sampleTester.matrixG,
                                                                                   sampleTester.matrixGprime, k))
            frobDistanceArr.append(sampleTester.aveFrobDistance)
        for l in range(len(specifiedKArr)):
            allAveNeighArr[l].append(sum(aveNeighArr[l]) / len(aveNeighArr[l]))
        aveFrobDistanceArr.append(sum(frobDistanceArr) / len(frobDistanceArr))

    if plotFrob:
        rankFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        GraphEstimates.plot_frob_error_against_rank_constraint(frobAx, rankConstraints, aveFrobDistanceArr)
    else:
        rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    GraphEstimates.plot_error_against_rank_constraint(neighAx, rankConstraints, allAveNeighArr, specifiedKArr)

def investigate_training_size(*, imageSet: NDArray, imageProductType: str, embeddingType:str, startingTrainingSize: int,
                              endingTrainingSize: int, increment=50, testSize: int,
                              testPrefix: str, specifiedKArr=None, plotFrob=True, trials=5, weight=""):
    """
    :param imageSet: Set of images used to the test and sample image sets. Currently, the training set takes from the front
    of the image set, and the test set takes from the tail of the image set TODO Do a proper monte carlo simulation.
    :param imageProductType: Image product type to investigate
    :param embeddingType: Embedding type to investigate
    :param startingTrainingSize: Starting point for the sweep (inclusive)
    :param endingTrainingSize: Ending point for the sweep
    :param increment: Increment in the sweep
    :param testSize: Size of the test set
    :param testPrefix: Prefix to the test names
    :param specifiedKArr: Specified k to plot for the neighbour graphs
    :param plotFrob: If true, also plot the frob error against sample size
    :return: Generates a graph for the k neighbour score against the size of the test sample.
    For now, training set is obtained from the start of imageSet and testing from the end of imageSet.
    IMPORTANT: Hence as the sample size increases, the training set remains mostly the same, and new images are added
    """
    if startingTrainingSize > endingTrainingSize:
        raise ValueError("Starting sample size must be lower than ending")
    sampleSizeArr = list(range(startingTrainingSize, endingTrainingSize, increment))
    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[] for i in specifiedKArr]
    aveFrobDistanceArr = []
    for sampleSizeTested in sampleSizeArr:
        logging.info("Investigating sample size " + str(sampleSizeTested) + " of " + str(endingTrainingSize))

        sampleName = testPrefix + "_sample_" + str(sampleSizeTested) + " of " + str(endingTrainingSize)
        testName = testPrefix + "_test_" + str(sampleSizeTested) + " of " + str(endingTrainingSize)
        frobDistanceArr = []
        aveNeighArr = [[] for i in specifiedKArr]

        # TODO Start Random Sampling here
        for j in range(trials):
            # Taking random training and testing samples
            testSample, trainingSample = generate_random_sample(imageSet, testSize)

            sampleEstimator = SampleEstimator(sampleName=sampleName + "_" + str(j), trainingImageSet=trainingSample,
                                            embeddingType=embeddingType, imageProductType=imageProductType, weight=weight)
            sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)

            # For each value of k, add the result to the array
            for i in range(len(specifiedKArr)):
                k = specifiedKArr[i]
                aveNeighArr[i].append(metrics.get_mean_normed_k_neighbour_score(sampleTester.matrixG,
                                                                               sampleTester.matrixGprime, k))
            frobDistanceArr.append(sampleTester.aveFrobDistance)
        for l in range(len(specifiedKArr)):
            allAveNeighArr[l].append(sum(aveNeighArr[l]) / len(aveNeighArr[l]))
        aveFrobDistanceArr.append(sum(frobDistanceArr) / len(frobDistanceArr))

    if plotFrob:
        trainingFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        GraphEstimates.plot_frob_error_against_training_size(frobAx, sampleSizeArr, aveFrobDistanceArr)
    else:
        trainingFig, neighAx = plt.subplots(1, len(specifiedKArr))
    GraphEstimates.plot_error_against_sample_size(neighAx, sampleSizeArr, allAveNeighArr, specifiedKArr)

def generate_random_sample(imageSet: NDArray, testSampleSize: int, trainingSampleSize: int):
    if testSampleSize + trainingSampleSize > len(imageSet):
        raise ValueError("Training and test sample size too large! Use size less than " + str(len(imageSet)) + ".")
    rng = np.random.default_rng()
    trainingSample = rng.choice(imageSet, trainingSampleSize, replace=False)
    remaining = np.asarray([image for image in imageSet.tolist() if image not in trainingSample.tolist()])
    testSample = rng.choice(remaining, testSampleSize, replace=False)
    return testSample, trainingSample

def investigate_training_size_for_image_products(*, imageSet: NDArray, imageProductTypes: list, embeddingType:str,
                                                 startingTrainingSize: int, endingTrainingSize: int, increment=50,
                                                 testSize: int, testPrefix: str, specifiedKArr=None, plotFrob=False,
                                                 weights=None, trials=5):
    """
    :param imageSet: Set of images used to the test and sample image sets. Currently, the training set takes from the front
    of the image set, and the test set takes from the tail of the image set TODO Do a proper monte carlo simulation.
    :param imageProductTypes: Image product types to investigate
    :param embeddingType: Embedding type to investigate
    :param startingTrainingSize: Starting point for the sweep (inclusive)
    :param endingTrainingSize: Ending point for the sweep
    :param increment: Increment in the sweep
    :param testSize: Size of the test set
    :param testPrefix: Prefix to the test names
    :param specifiedKArr: Specified k to plot for the neighbour graphs
    :param plotFrob: If true, also plot the frob error against sample size
    :return: Generates a graph for the k neighbour score against the size of the test sample.
    For now, training set is obtained from the start of imageSet and testing from the end of imageSet.
    IMPORTANT: Hence as the sample size increases, the training set remains mostly the same, and new images are added
    """
    if startingTrainingSize > endingTrainingSize:
        raise ValueError("Starting sample size must be lower than ending")
    sampleSizeArr = list(range(startingTrainingSize, endingTrainingSize, increment))
    if weights is None:
        weights = ["" for image_product_type in imageProductTypes]
    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[[] for imageProductType in imageProductTypes] for i in specifiedKArr]
    aveFrobDistanceArr = [[] for imageProductType in imageProductTypes]
    for sampleSizeTested in sampleSizeArr:
        logging.info("Investigating sample size " + str(sampleSizeTested) + " of " + str(endingTrainingSize))

        sampleName = testPrefix + "_sample_" + str(sampleSizeTested) + " of " + str(endingTrainingSize)
        testName = testPrefix + "_test_" + str(sampleSizeTested) + " of " + str(endingTrainingSize)
        # TODO Start Random Sampling here
        frobDistanceArr = [[] for imageProductType in imageProductTypes]
        aveNeighArr = [[[] for imageProductType in imageProductTypes] for i in specifiedKArr]

        for index in range(len(imageProductTypes)):
            imageProductType = imageProductTypes[index]
            weight = weights[index]
            # TODO Start Random Sampling here
            for j in range(trials):
                # Taking random training and testing samples
                testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSizeTested)

                sampleEstimator = SampleEstimator(sampleName=sampleName + "_" + str(j), trainingImageSet=trainingSample,
                                                  embeddingType=embeddingType, imageProductType=imageProductType, weight=weight)
                sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)

                # For each value of k, add the result to the array
                for i in range(len(specifiedKArr)):
                    k = specifiedKArr[i]
                    aveNeighArr[i][index].append(metrics.get_mean_normed_k_neighbour_score(sampleTester.matrixG,
                                                                                    sampleTester.matrixGprime, k))
                frobDistanceArr[index].append(sampleTester.aveFrobDistance)
            for l in range(len(specifiedKArr)):
                allAveNeighArr[l][index].append(sum(aveNeighArr[l][index]) / len(aveNeighArr[l][index]))
            aveFrobDistanceArr[index].append(sum(frobDistanceArr[index]) / len(frobDistanceArr[index]))

    if plotFrob:
        trainingFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        #GraphEstimates.plot_frob_error_against_training_size(frobAx, sampleSizeArr, aveFrobDistanceArr)
    else:
        trainingFig, neighAx = plt.subplots(1, len(specifiedKArr))
    if type(neighAx) is not list:
        neighAx = [neighAx]
    GraphEstimates.plot_error_against_sample_size_for_image_types(neighAx, sampleSizeArr, allAveNeighArr, specifiedKArr,
                                                                  imageProductTypes, weights)

def investigate_tester_rank_constraint_for_image_products(*, imageSet: NDArray, imageProductTypes: list, sampleSize: int,
                                                          testSize: int, testPrefix: str, startingConstr: int,
                                                          endingConstr: int, increment=1, specifiedKArr=None,
                                                          plotFrob=False, trials=5, weights=None, progressive=False,
                                                          embeddings=None):
    """
    :param specifiedKArr: value of k for the k neighbour score
    :param imageSet: Set of images used to the test and sample image sets. Currently, the training set takes from the front
    of the image set, and the test set takes from the tail of the image set TODO Do a proper monte carlo simulation.
    :param imageProductTypes: Image product types to investigate
    :param startingConstr: Starting lowest rank constraint to start the sweep inclusive
    :param endingConstr: Final largest rank constraint to end the sweep inclusive
    :param increment: Increment in the rank constraint sweep
    :param testPrefix: Used as a prefix to all the test names
    :param testSize: Size of the test set
    :param sampleSize: Size of the sample set
    :param plotFrob: If True, also plots frob error against rank
    :return: Uses the penncorr method to generate embeddings for different rank constraints
    Makes a graph of the average neighbour score against rank_constraint and
    average frobenius distance against rank_constraint
    Remember to use plt.show() to display plots

    Aims to answer the question: How does the rank constraint affect the error of the embeddings generated by penncorr?
    """
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]
    if weights is None:
        weights = ["" for image_product_type in imageProductTypes]
    if embeddings is None:
        embeddings = ["pencorr" for image_product_type in imageProductTypes]

    aveFrobDistanceArr = [[] for imageProductType in imageProductTypes]
    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[[] for imageProductType in imageProductTypes] for i in specifiedKArr]
    rankConstraints = list(range(startingConstr, endingConstr + 1, increment))
    if progressive:
        rankConstraints = metrics.get_progressive_range(startingConstr, endingConstr + 1, increment)


    # For each rank in the sweep, generate a SampleTester and add its results to the array
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + " of " + str(endingConstr))

        for index in range(len(imageProductTypes)):
            imageProductType = imageProductTypes[index]
            embType = embeddings[index] + "_" + str(rank)
            weight = weights[index]
            sampleName = testPrefix + "_sample_" + str(rank) + " of " + str(endingConstr)
            testName = testPrefix + "_test_" + str(rank) + " of " + str(endingConstr)

            aveNeighArr = [[] for i in specifiedKArr]
            frobDistanceArr = []
            # TODO Start Random Sampling Here
            for j in range(trials):
                # Taking training and testing samples as random samples of the image set
                testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSize)

                # Generating a sampleEstimator and SampleTester with the input parameters
                sampleEstimator = SampleEstimator(sampleName=sampleName + "_sample_" + str(j), trainingImageSet=trainingSample, embeddingType=embType,
                                                  imageProductType=imageProductType, weight=weight)
                sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)


                # For each k to be investigated, append the respective k neighbour score
                for i in range(len(specifiedKArr)):
                    k = specifiedKArr[i]
                    aveNeighArr[i].append(metrics.get_mean_normed_k_neighbour_score(sampleTester.matrixG,
                                                                                    sampleTester.matrixGprime, k))
                frobDistanceArr.append(sampleTester.aveFrobDistance)
            for l in range(len(specifiedKArr)):
                allAveNeighArr[l][index].append(sum(aveNeighArr[l]) / len(aveNeighArr[l]))
            aveFrobDistanceArr[index].append(sum(frobDistanceArr) / len(frobDistanceArr))

    if plotFrob:
        rankFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        GraphEstimates.plot_frob_error_against_rank_constraint(frobAx, rankConstraints, aveFrobDistanceArr)
    else:
        rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    if type(neighAx) is not list:
        neighAx = [neighAx]
    GraphEstimates.plot_error_against_rank_constraint_for_image_products(neighAx, rankConstraints, allAveNeighArr, specifiedKArr,
                                                                         imageProducts=imageProductTypes, weights=weights)

def investigate_sample_and_test_sets(*, trainingSet: str, testSet: str, trainingSize: int, testSize: int,
                                     imageProductTypes: list, testPrefix: str,
                                     startingConstr: int, endingConstr: int, increment=1, specifiedKArr=None, trials=5,
                                     weights=None, progressive=False, embeddings=None, filters=None, plotFrob=False):
    if startingConstr >= endingConstr:
        raise ValueError("Starting rank constraint must be lower than ending constraint")
    if specifiedKArr is None:
        specifiedKArr = [5]
    if weights is None:
        weights = ["" for image_product_type in imageProductTypes]
    if embeddings is None:
        embeddings = ["pencorr" for image_product_type in imageProductTypes]
    if filters is None:
        filters = []

    training_set_filepath = fputils.get_image_set_filepath(trainingSet, filters)
    full_training_image_set = utils.generate_filtered_image_set(trainingSet, filters, training_set_filepath)
    sameSet = False

    if trainingSet == testSet:
        sameSet = True
    else:
        test_set_filepath = fputils.get_image_set_filepath(testSet, filters)
        full_test_image_set = utils.generate_filtered_image_set(testSet, filters, test_set_filepath)


    aveFrobDistanceArr = [[] for imageProductType in imageProductTypes]
    # A list of k neighbour plotting data, for each of the k in specified K array
    allAveNeighArr = [[[] for imageProductType in imageProductTypes] for i in specifiedKArr]
    rankConstraints = list(range(startingConstr, endingConstr + 1, increment))
    if progressive:
        rankConstraints = metrics.get_progressive_range(startingConstr, endingConstr + 1, increment)


    # For each rank in the sweep, generate a SampleTester and add its results to the array
    for rank in rankConstraints:
        logging.info("Investigating rank " + str(rank) + " of " + str(endingConstr))

        for index in range(len(imageProductTypes)):
            imageProductType = imageProductTypes[index]
            embType = embeddings[index] + "_" + str(rank)
            weight = weights[index]
            sampleName = testPrefix + "_sample_" + str(rank) + " of " + str(endingConstr)
            testName = testPrefix + "_test_" + str(rank) + " of " + str(endingConstr)

            aveNeighArr = [[] for i in specifiedKArr]
            frobDistanceArr = []
            # TODO Start Random Sampling Here
            for j in range(trials):
                # Taking training and testing samples as random samples of the image set
                if sameSet:
                    testSample, trainingSample = generate_random_sample(full_training_image_set, testSize, trainingSize)
                else:
                    disregard, trainingSample = generate_random_sample(full_training_image_set, 0, trainingSize)
                    testSample, disregard = generate_random_sample(full_test_image_set, testSize, 0)

                # Generating a sampleEstimator and SampleTester with the input parameters
                sampleEstimator = SampleEstimator(sampleName=sampleName + "_sample_" + str(j), trainingImageSet=trainingSample, embeddingType=embType,
                                                  imageProductType=imageProductType, weight=weight)
                sampleTester = SampleTester(testImages=testSample, sampleEstimator=sampleEstimator, testName=testName)


                # For each k to be investigated, append the respective k neighbour score
                for i in range(len(specifiedKArr)):
                    k = specifiedKArr[i]
                    aveNeighArr[i].append(metrics.get_mean_normed_k_neighbour_score(sampleTester.matrixG,
                                                                                    sampleTester.matrixGprime, k))
                frobDistanceArr.append(sampleTester.aveFrobDistance)
            for l in range(len(specifiedKArr)):
                allAveNeighArr[l][index].append(sum(aveNeighArr[l]) / len(aveNeighArr[l]))
            aveFrobDistanceArr[index].append(sum(frobDistanceArr) / len(frobDistanceArr))

    if plotFrob:
        rankFig, axArr = plt.subplots(1, len(specifiedKArr) + 1)
        frobAx = axArr[-1]
        neighAx = axArr[:-1]
        GraphEstimates.plot_frob_error_against_rank_constraint(frobAx, rankConstraints, aveFrobDistanceArr)
    else:
        rankFig, neighAx = plt.subplots(1, len(specifiedKArr))
    if type(neighAx) is not list:
        neighAx = [neighAx]
    GraphEstimates.plot_error_against_rank_constraint_for_image_products(neighAx, rankConstraints, allAveNeighArr, specifiedKArr,
                                                                         imageProducts=imageProductTypes, weights=weights)
