# I made this as a separate file, so that we can
# experiment with our algorithm in assignment1.py but always
# have a working version here
# We should think about how to decide whether the algorithm converged
# I now decided to count a error rate of 0 after all epochs as success
# maybe we can stop the algorithm early though.

import numpy as np
# from matplotlib import pyplot as plt


def calculateError(weights, data):
    errors = 0
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        currentPoint = data[iter][0:2].reshape(1, 2)
        currentLabel = data[iter][2]
        localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
        if np.sign(localPotential)!=np.sign(currentLabel):
            errors = errors + 1

    return 1-errors/numberOfPoints

def classify(data, epochs, learning_rate):
    for i in range(epochs):
        numberOfPoints = data.shape[0]
        for iter in range(numberOfPoints):
            #print("epoch " + str(i))
            #print("iteration " + str(iter))
            # print("weights " + str(weights))
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:2].reshape(1, 2)
            currentLabel = data[iter][2]

            if iter == 0 and i == 0:
                # initially assign the weights as the first point
                weights = (np.multiply([[1, 1]], currentPoint) * data[iter][2]) * learning_rate
            else:
                # Assign the weights as the dot product
                localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
                # print("local Potential "+ str(localPotential))
                # Since the cutoff point is zero we check whether the local potential is smaller or equal than 0
                # only then the hebbian term becomes 1 and the weights are updated
                if localPotential > 0:
                    weights = weights  # hebbian = 0
                else:
                    weights = weights + learning_rate * currentPoint * currentLabel

                # updating the weights
                # weights = weights + learnRate * hebbian * currentPoint * currentLabel

            # print(weights)
    if calculateError(weights, data) == 0:
        return 1 # success, perfect classification
    else:
        return 0 # did not converge or classification not possible