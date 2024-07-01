import itertools
import re

import numpy as np
from numpy.typing import NDArray
from skimage.draw import polygon

from src.helpers.IslandCreator import grid_creation



def get_binary_image_set(imageLength: int, maxOnesPercentage=100) -> NDArray[int]:
    if maxOnesPercentage > 100:
        raise ValueError(str(maxOnesPercentage) + " > 100. Maximum percentage of ones has to be less than 100")
    cutoff = maxOnesPercentage * (imageLength ** 2) // 100
    fullList = []
    for item in itertools.product([0, 1], repeat=imageLength ** 2):
        if np.sum(item) <= cutoff:
            fullList.append(np.reshape(item, (imageLength, imageLength)))
    fullList = np.array(fullList)
    return fullList


def get_image_set(imageType: str):
    if re.search('[0-9]?[0-9]bin[0-9]?[0-9]max_ones$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        imageLength = int(re.search(r'^\d+', imageType).group())
        maxOnesPercentage = int(re.search(r'\d+', imageType[2:]).group())
        image_set = get_binary_image_set(imageLength, maxOnesPercentage=maxOnesPercentage)
    elif re.search('[0-9]?[0-9]bin$', imageType) is not None:  # Searching if the image type follows the format 2bin
        imageLength = int(re.search(r'\d+', imageType).group())
        image_set = get_binary_image_set(imageLength)
    elif imageType == "triangles":
        image_set = get_triangles_image_set()
    elif imageType == "quadrilaterals":
        image_set = get_quadrilaterals_image_set()
    elif re.search(r'shapes', imageType) is not None:
        size, border_size, sides = parse_shapes_set(imageType)
        image_set = []
        for j in sides:
            image_set.extend(get_shapes_set(size, j, border_size).tolist())
        image_set = np.array(image_set)
    elif re.search('[0-9]?[0-9]island[0-9]?[0-9]max_ones[0-9]?[0-9]images$', imageType) is not None:  # Searching if image type follows the
        # format of 3bin40max_ones
        matches = re.findall(r"\d", imageType)
        imageLength = int(matches[0])
        maxOnesPercentage = int(matches[1])
        numImages = int(matches[2])
        image_set = get_island_image_set(imageLength, maxOnesPercentage, numImages)
    else:
        raise ValueError(imageType + " is not a valid image type")
    return image_set


def get_island_image_set(imageLength, maxOnesPercentage, numImages):
    """
    :param imageLength: side length of image grid
    :param maxOnesPercentage: Percent of ones
    :param numImages: number of images to generate
    :return: An image set of randomly generated islands with no repeats
    """
    return np.array(grid_creation(imageLength, numImages, int(maxOnesPercentage / 100 * (imageLength ** 2))))


def get_triangles_image_set():
    """
    :return: The image set of 4x4 triangles within an 8x8 matrix
    """
    return get_shapes_set(4, 3, 2)

def get_quadrilaterals_image_set():
    return get_shapes_set(4, 4, 2)

def parse_shapes_set(imageType: str):
    """
    Parses the imageType string for an image set containing multiple shapes
    :param imageType: imageType string
    :return: Parameters for the whole image set
    """
    params = imageType.split("_")
    dims = False
    size = 4
    border_size = 2
    sides = []
    for i in range(1, len(params)):
        if params[i] == "dims":
            dims = True
            continue
        elif dims:
            size = int(params[i])
            border_size = int(params[i + 1])
            break
        sides.append(int(params[i]))
    return size, border_size, sides

def get_shapes_set(size: int, sides: int, border_size: int):
    image_set = []
    indexes = list(range(0, size ** 2))
    for comb in itertools.permutations(indexes, r=sides):
        image = np.zeros((size, size), dtype=int)
        r = []
        c = []
        for index in comb:
            r.append(index // size)
            c.append(index % size)

        # Check for straight lines
        points = []
        for i in range(0, len(r)):
            points.append((r[i], c[i]))
        collinear = False
        for coords in itertools.combinations(points, 3):
            m1 = (coords[1][1] - coords[0][1]) * (coords[2][0] - coords[1][0])
            m2 = (coords[2][1] - coords[1][1]) * (coords[1][0] - coords[0][0])
            if m1 == m2:
                collinear = True
                break
        if collinear:
            continue

        # Check for intersecting lines within shape
        if sides > 3:
            intersect = False
            lines = []
            for i in range(0, sides):
                start = (r[i % sides], c[i % sides])
                end = (r[(i + 1) % sides], c[(i + 1) % sides])
                lines.append((start, end))
            for pair in itertools.combinations(lines, 2):
                test = is_intersecting(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
                if test:
                    intersect = test
                    break
            if intersect:
                continue

        # Create shape
        rr, cc = polygon(r, c)
        image[rr, cc] = 1
        image = np.pad(image, (border_size, border_size), constant_values=(0, 0))
        image_set.append(image)
    image_set = np.array(image_set)
    image_set = np.unique(image_set, axis=0)
    return image_set

def cross_test(p1, p2, p3):
    """
    :param p1: Point 1
    :param p2: Point 2
    :param p3: Point 3
    :return: Tests gradient for checking intersections
    """
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_intersecting(p1, p2, p3, p4):
    """
    Checks if two line segments created using the 4 points intersect in the middle
    :param p1: Point 1
    :param p2: Point 2
    :param p3: Point 3
    :param p4: Point 4
    :return: True if the two lines are intersecting, false if not
    """
    # Check if the segments share a point. If so, they do not intersect in the middle
    if p1 == p4 or p2 == p3:
        return False

    return cross_test(p1, p3, p4) != cross_test(p2, p3, p4) and cross_test(p1, p2, p3) != cross_test(p1, p2, p4)
