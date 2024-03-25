import os.path
import random
from statistics import median
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy._typing import NDArray
from src.data_processing.BruteForceEstimator import BruteForceEstimator
from data_processing import FilepathUtils
import visualization.Metrics as metrics


def plot_eigenvalues(ax1: Axes, ax2: Axes, initialEigenvalues: NDArray, finalEigenvalues: NDArray):
    """
    :param ax1: axes to plot the largest 20%/top 15 eigenvalues
    :param ax2: axes to plot the lowest 20%/bottom 15 eigenvalues
    :param initialEigenvalues: initial eigenvlaues of the image product matrix
    :param finalEigenvalues: final eigenvalues of the embedding matrix dot product
    :return:
    """
    barWidth = 0.4

    numPlot = min(int(len(initialEigenvalues) * 0.2), 15)

    topInitEigen = initialEigenvalues[:numPlot]
    topFinalEigen = finalEigenvalues[:numPlot]
    # Set position of bar on X axis
    br1 = np.arange(numPlot)
    br2 = [x + barWidth for x in br1]
    ax1.set_title("Top " + str(numPlot) + " eigenvalues")
    rects1 = ax1.bar(br1, topInitEigen, color='r', width=barWidth, label='IT')
    rects2 = ax1.bar(br2, topFinalEigen, color='g', width=barWidth, label='ECE')
    ax1.legend((rects1[0], rects2[0]), ('Initial Eigenvalues', 'Final eigenvalues'))

    bottomInitEigen = initialEigenvalues[-numPlot:]
    bottomFinalEigen = finalEigenvalues[-numPlot:]

    ax2.set_title("Bottom " + str(numPlot) + " eigenvalues")
    rects1 = ax2.bar(br1, bottomInitEigen, color='r', width=barWidth, label='IT')
    rects2 = ax2.bar(br2, bottomFinalEigen, color='g', width=barWidth, label='ECE')
    ax2.legend((rects1[0], rects2[0]), ('Initial Eigenvalues', 'Final eigenvalues'))


def plot_ave_k_neighbours(ax, allAveKNeighbourScores: List, kArr: List):
    """
    :param ax: Axes to plot the graph
    :param allAveKNeighbourScores: The list af all ave k neighbour scores for all values of k
    :param kArr: List of k values to plot
    :return: Plots the graph of average k neighbour score for a range of k values
    """
    idealPlot = range(1, len(kArr) + 1)  # for plotting ideal score
    ax.plot(idealPlot, [1 for count in range(len(idealPlot))], color='b', linestyle=':', label="Ideal")
    ax.plot(kArr, allAveKNeighbourScores, color='r', label="Real")
    ax.set_title("Mean neighbour score against number of neighbours analysed")
    ax.set_xlabel("Value of k")
    ax.set_ylabel("Norm K neighbour score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")


def plot_k_histogram(ax: Axes, kNeighScores: List, kVal: int):
    labels, counts = np.unique(kNeighScores, return_counts=True)
    ax.set_xlim(-0.5, kVal + 0.5)
    ax.bar(labels, counts, align='center')
    ax.set_xticks(labels)
    ax.set_title("Histogram of K scores for k = " + str(kVal), fontsize=12)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Value of K score")

def plot_single_image_k_neighbours(imageAx: Axes, neighAx: Axes, image: NDArray, imageTitle:str, kArr: List[float], imageKNeighScore: List[float]):
    """
    :param imageAx: Axes to plot the image itself
    :param neighAx: Axes to plot the nieghbour graph for the image
    :param image: 2d numpy array of pixels
    :param imageTitle: Title of the image
    :param imageKNeighScore: K neighbour score of the image
    :return:
    """
    imageAx.set_title(imageTitle)
    imageAx.imshow(image, cmap='Greys', interpolation='nearest')

    # Ideal plot:
    neighAx.plot(kArr, [1 for count in range(len(kArr))], color='b', linestyle=':', label="Ideal")
    neighAx.plot(kArr, imageKNeighScore, color='r', label="Real")
    neighAx.set_title(
        "Normed k neighbour score of " + imageTitle + " against k", fontsize=12)

    neighAx.set_xlabel("k")
    neighAx.set_ylabel("Norm K neigh score", fontsize=8)
    neighAx.set_ylim(0, 1.1)
    neighAx.legend(loc="lower right")


def plot_key_stats_text(ax: Axes, bfEstimator: BruteForceEstimator):
    displayText = ("Frobenius norm of difference between imageProductMatrix and A^tA: " + "{:.2f}".format(
        bfEstimator.frobDistance) + "\n" +
                   "Average Frobenius norm of difference between imageProductMatrix and A^tA: " + "{:.3E}".format(
                bfEstimator.aveFrobDistance) + "\n" +
                   "Greatest single element difference between imageProductMatrix and A^tA: " + "{:.2f}".format(
                bfEstimator.maxDifference) + "\n")
    ax.text(0.5, 0.5, displayText, color='black',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')




def plot_error_against_sample_size(neighbourAxArr: List[Axes], sampleSizeArr: List, fullNeighArr: List,
                                       specifiedKArr: List):

    for count in range(len(specifiedKArr)):
        neighbourAx = neighbourAxArr[count]
        specifiedK = specifiedKArr[count]
        neighArr = [arr[count] for arr in fullNeighArr]

        # for plotting the max possible score
        neighbourAx.plot(sampleSizeArr, [1 for i in range(len(neighArr))], color='b', linestyle=':', label="Ideal")
        neighbourAx.plot(sampleSizeArr, neighArr, color='r', label="Real")
        neighbourAx.set_title(
            "Mean norm k neighbour score against sample size (k = " + str(
                specifiedK) + ")")
        neighbourAx.set_xlabel("Sample size")
        neighbourAx.set_ylabel("Mean K neighbour score (k = " + str(specifiedK) + ")")
        neighbourAx.set_ylim(0, 1.05)
        neighbourAx.legend(loc="upper left")


def plot_frob_error_against_rank_constraint(frobAx: Axes, rankArr: List[int], frobArr: List[float]):
    frobAx.plot(rankArr, frobArr)
    frobAx.set_title("Average frobenius error against rank constraint")
    frobAx.set_xlabel("Rank Constraint")
    frobAx.set_ylabel("Average frobenius error")

def plot_error_against_rank_constraint(neighbourAxArr: List[Axes], rankArr: List, fullNeighArr: List,
                                       specifiedKArr: List):
    """
    :param neighbourAxArr: Axes to plot the neighbour graph
    :param rankArr: array of rank constrain values to plot (x axis for both graphs)
    :param fullNeighArr: List of all the data for the neighbour graphs (y axis)
    :param specifiedKArr: The list k neighbour scores to be used
    :return:
    """

    for count in range(len(specifiedKArr)):
        neighbourAx = neighbourAxArr[count]
        specifiedK = specifiedKArr[count]
        neighArr = fullNeighArr[count]

        idealPlot = [1 for i in range(len(neighArr))]  # for plotting the max possible score
        neighbourAx.plot(rankArr, idealPlot, color='b', linestyle=':', label="Ideal")
        neighbourAx.plot(rankArr, neighArr, color='r', label="Real")
        neighbourAx.set_title(
            "Mean norm k neighbour score against the rank constraint (k = " + str(
                specifiedK) + ")")
        neighbourAx.set_xlabel("Rank Constraint")
        neighbourAx.set_ylabel("Mean K neighbour score (k = " + str(specifiedK) + ")")
        neighbourAx.set_ylim(0, 1.05)
        neighbourAx.legend(loc="lower right")
