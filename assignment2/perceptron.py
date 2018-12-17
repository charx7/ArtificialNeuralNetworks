# I made this as a separate file, so that we can
# experiment with our algorithm in assignment1.py but always
# have a working version here
# We should think about how to decide whether the algorithm converged
# I now decided to count a error rate of 0 after all epochs as success
# maybe we can stop the algorithm early though.

import numpy as np
# from matplotlib import pyplot as plt

# TODO determine the generalization error based on the randomly generated "teacher perceptron"
def calculateAccuracy(weights, data, dims):
    errors = 0
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        currentPoint = data[iter][0:dims].reshape(1, dims)
        currentLabel = data[iter][dims]
        # localPotential = np.dot(weights.flatten(), currentPoint.flatten())
        classification = np.sign(np.dot(weights.flatten(), currentPoint.flatten()))
        # print('The classification label is: ',classification)
        # print('The correct label is: ', np.sign(currentLabel))
        if classification != np.sign(currentLabel):
             errors = errors + 1

    return 1-errors/numberOfPoints

def classify(data, epochs, learning_rate, dims):
    # Init weights at 0
    weights = np.zeros(dims)
    epsilon = 0.01

    for i in range(epochs):
        weights_before = weights
        weights_after = 10
        numberOfPoints = data.shape[0]
        # Start local potential vector
        # TODO change the stop criteria to a epsilon weights change one

        localPotentialVector = []
        for iter in range(numberOfPoints):
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:dims].reshape(1, dims)
            currentLabel = data[iter][dims]
            # Local potential Calculation
            localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
            # Update the local potential vector
            localPotentialVector.append(localPotential)
            # Check if the dot product is =< 0 to update (there was no need for the i=0 condition)
            # TODO the update should be using the minimum stability of the minover algo
            # HINT: calculate and store the kappas
            # Move to outside of this loop (need to calculate kappas for every data point first)
            if localPotential <= 0:
                weights = weights + learning_rate *  currentPoint * currentLabel

        # Check if local potential is > 0 for every point
        localPotentialFiltered = filter(lambda a: a > 0, localPotentialVector)
        lenghtOfLocalPotential = len(list(localPotentialFiltered))

        # TODO REMOVE
        # Condition for an early epoch stop
        if lenghtOfLocalPotential == numberOfPoints:
            # Exit the epoch loop
            break

    acc = calculateAccuracy(weights, data, dims)
    if acc == 1:
        return 1 # success, perfect classification
    else:
        return 0 # did not converge or classification not possible
