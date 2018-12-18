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

def doHebbianUpdate(point_min_kappa, label_min_kappa, learning_rate, old_weights):
    # Do the hebbian update
    new_weights = old_weights + (learning_rate* np.dot(point_min_kappa, label_min_kappa))
    return new_weights

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
        # Get the points that are going to be used for the hebbian update
        point_min_kappa = data[index_minKappa][0:dims]
        label_min_kappa = data[index_minKappa][dims]

        # Norm calc for the stop criteria
        old_weights_norm = np.linalg.norm(weights)
        # Call to the hebbian update function on the points
        weights = doHebbianUpdate(point_min_kappa, label_min_kappa, learning_rate, weights)
        # Norm calculation for the stop criteia
        new_weights_norm = np.linalg.norm(weights)
        # Stop criteia difference of the norms
        stop_criteria = abs(new_weights_norm - old_weights_norm)
        print('The stop criteria is: ', stop_criteria)
        if stop_criteria < 0.01:
            print('This should early stop!')
            # escape the epoch loop
            break

    acc = calculateAccuracy(weights, data, dims)
    if acc == 1:
        return 1 # success, perfect classification
    else:
        return 0 # did not converge or classification not possible
