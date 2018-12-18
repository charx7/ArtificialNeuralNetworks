import numpy as np

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
    # get the weights as the first data point (special case for the first)
    weights = data[0][0:dims]

    for i in range(epochs):
        numberOfPoints = data.shape[0]
        # Start local potential vector
        # TODO change the stop criteria to a epsilon weights change one

        # Stabilities vector
        kappasVector = []
        for iter in range(numberOfPoints):
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:dims]#.reshape(1, dims)
            currentLabel = data[iter][dims]

            # Calculate Kappa
            # currently division by 0 problem
            currentKappa = np.dot(weights, currentPoint * currentLabel) / np.linalg.norm(weights)

            # append an stability(kappa) dictionary
            kappasVector.append(currentKappa)

        # Get the min kappa
        minKappa = min(kappasVector)
        index_minKappa = np.argmin(kappasVector)
        #print('\nThe index of min kappa is: \n', index_minKappa)
        # Do hebbian update

    acc = calculateAccuracy(weights, data, dims)
    if acc == 1:
        return 1 # success, perfect classification
    else:
        return 0 # did not converge or classification not possible
