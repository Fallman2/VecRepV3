import logging
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
import pandas as pd

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
                                       specifiedKArr=None, plotFrob=True, weight="", trials=5):
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
        for j in range(trials):
            # Taking training and testing samples as random samples of the image set
            testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSize, seed=j)

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
            testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSizeTested, seed=j)

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

def generate_random_sample(imageSet: NDArray, testSampleSize: int, trainingSampleSize: int, seed=0):
    if testSampleSize + trainingSampleSize > len(imageSet):
        raise ValueError("Training and test sample size too large! Use size less than " + str(len(imageSet)) + ".")
    rng = np.random.default_rng(seed=500 + seed)
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
                testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSizeTested, seed=j)

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
                testSample, trainingSample = generate_random_sample(imageSet, testSize, sampleSize, seed=j)

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

    if trainingSet == testSet:
        sameSet = True
    else:
        sameSet = False
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
                    testSample, trainingSample = generate_random_sample(full_training_image_set, testSize, trainingSize, seed=j)
                else:
                    disregard, trainingSample = generate_random_sample(full_training_image_set, 0, trainingSize, seed=j)
                    testSample, disregard = generate_random_sample(full_test_image_set, testSize, 0, seed=j)

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


def investigate_sample_plateau_rank(*, training_sets: list, test_sets: list, training_sizes: list,
                                    image_product_types: list, test_prefix: str,
                                    k=5, trials=5, prox=3,
                                    weights=None, embeddings=None, filters=None):

    if weights is None:
        weights = ["" for image_product_type in image_product_types]
    if embeddings is None:
        embeddings = ["pencorr_python" for image_product_type in image_product_types]
    if filters is None:
        filters = []

    data = {"Training Set": training_sets, "Test Set": test_sets, "Image Size": [], "Image Products": image_product_types,
            "Embeddings": embeddings, "Weights": weights,
            "K_scores": [], "Training Set Size": [], "Non_zero": [], "Plateau Rank": []}

    for index, trainingSet in enumerate(training_sets):
        testSet = test_sets[index]
        embedding = embeddings[index]
        weight = weights[index]
        image_product = image_product_types[index]
        training_size = training_sizes[index]
        ave_neigh_arr = []
        average_plateau_rank_arr = []
        avg_non_zero = []

        training_set_filepath = fputils.get_image_set_filepath(trainingSet, filters)
        full_training_image_set = utils.generate_filtered_image_set(trainingSet, filters, training_set_filepath)

        test_set_filepath = fputils.get_image_set_filepath(testSet, filters)
        test_sample = utils.generate_filtered_image_set(testSet, filters, test_set_filepath)

        image_size = len(full_training_image_set[0])
        data["Image Size"].append(image_size)

        for i in range(trials):
            disregard, training_sample = generate_random_sample(full_training_image_set, 0, training_size, seed=i)

            # Loop variables
            high = training_size
            low = 0
            selected_rank = high
            max_k_score = 2
            iterations = 0
            same_rank = 0
            score_change = False
            max_score_rank = []

            while high - low > prox:
                logging.info("Starting iteration " + str(iterations + 1))
                selected_embedding = embedding + "_" + str(selected_rank)
                sampleName = test_prefix + "_" + trainingSet + "_constraint_" + str(selected_rank)
                sampleEstimator = SampleEstimator(sampleName=sampleName + "_sample_" + str(i),
                                                  trainingImageSet=training_sample, embeddingType=selected_embedding,
                                                  imageProductType=image_product, weight=weight)
                testName = test_prefix + "_test"
                sample_tester = SampleTester(testImages=test_sample, sampleEstimator=sampleEstimator, testName=testName)

                k_score = metrics.get_mean_normed_k_neighbour_score(sample_tester.matrixG, sample_tester.matrixGprime, k)
                if not score_change:
                    # k_score is the same as previous tested rank constraint
                    if iterations == 0:
                        # First iteration, no rank constraint placed
                        max_k_score = k_score  # k_score value where plateau occurs
                        ave_neigh_arr.append(max_k_score)
                        nonzero = np.count_nonzero(np.array([np.max(b) - np.min(b) for b in sampleEstimator.embeddingMatrix]))
                        avg_non_zero.append(nonzero)  # Number of nonzero eigenvalues after pencorr acts as upper
                        high = training_size
                        low = training_size // 2
                    elif k_score == max_k_score:
                        # Not first iteration, k_score has yet to change. Continue lowering rank constraint.
                        high = low
                        low = high // 2
                    else:
                        # Not first iteration, k_score has changed. Begin looking for plateau rank.
                        score_change = True
                        low = ((high - low) // 2) + low
                elif k_score != max_k_score:
                    # Before plateau area. Raise rank constraint.
                    low = ((high - low) // 2) + low
                    max_score_rank = []
                    same_rank = 0
                elif same_rank == 2:
                    # Successively within plateau area. Break loop
                    high = max_score_rank[0]
                    iterations += 1
                    logging.info("Finishing iteration" + str(iterations))
                    break
                else:
                    # Within plateau area. Raise rank constraint slowly in case plateau rank not yet reached.
                    max_score_rank.append(low)
                    diff = (high - low) // 4
                    low += diff
                    same_rank += 1

                # Test next iteration at low estimate
                selected_rank = low
                iterations += 1
                logging.info("Finishing iteration" + str(iterations))
                logging.info("Next Rank " + str(low))

            # Once loop ends, save high estimate plateau rank for currently tested image set
            logging.info("Plateau rank " + str(high))
            average_plateau_rank_arr.append(high)
        data["K_scores"].append(sum(ave_neigh_arr) / len(ave_neigh_arr))
        data["Plateau Rank"].append(sum(average_plateau_rank_arr) / len(average_plateau_rank_arr))
        data["Non_zero"].append(sum(avg_non_zero) / len(avg_non_zero))
    df = pd.DataFrame(data)
    return df


