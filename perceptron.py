# I made this as a separate file, so that we can
# experiment with our algorithm in assignment1.py but always
# have a working version here
# We should think about how to decide whether the algorithm converged
# I now decided to count a error rate of 0 after all epochs as success
# maybe we can stop the algorithm early though.

import numpy as np
# from matplotlib import pyplot as plt


def calculateError(weights, data, dims):
    errors = 0
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        currentPoint = data[iter][0:dims].reshape(1, dims)
        currentLabel = data[iter][dims]
        localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
        if np.sign(localPotential)!=np.sign(currentLabel):
            errors = errors + 1

    return 1-errors/numberOfPoints

def classify(data, epochs, learning_rate, dims):
    # Init weights at 0
    weights = np.zeros(dims)

    for i in range(epochs):
        numberOfPoints = data.shape[0]
        for iter in range(numberOfPoints):
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:dims].reshape(1, dims)
            currentLabel = data[iter][dims]
            # Local potential Calculation
            localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
            # Check if the dot product is =< 0 to update (there was no need for the i=0 condition)
            if localPotential <= 0:
                weights = weights + learning_rate *  currentPoint * currentLabel

    if calculateError(weights, data, dims) == 0:
        return 1 # success, perfect classification
    else:
        return 0 # did not converge or classification not possible
