from generateRandomData import generateRandomData
from testDataGeneration import plotData
import numpy as np
from matplotlib import pyplot as plt
# X datapoints in Y dimenions
data, weights = generateRandomData(3,2)
epochs = 3 # Number of epochs
learnRate = 1 / 2
for i in range(epochs):
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        print("epoch " + str(i))
        print("iteration " + str(iter))
        print("weights " + str(weights))
        # Get the current point and reshape it to be able to use the dot-product
        currentPoint = data[iter][0:2].reshape(1, 2)
        currentLabel = data[iter][2]

        if iter == 0 and i == 0:
            # initially assign the weights as the first point
            weights = (np.multiply([[1,1]], currentPoint) * data[iter][2] ) * learnRate
        else:
            # Assign the weights as the dot product
            localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
            print("local Potential "+ str(localPotential))
            # Since the cutoff point is zero we check whether the local potential is smaller or equal than 0
            # only then the hebbian term becomes 1 and the weights are updated
            if localPotential > 0:
                weights = weights # hebbian = 0
            else:
                weights = weights + learnRate * currentPoint * currentLabel

            #updating the weights
            # weights = weights + learnRate * hebbian * currentPoint * currentLabel

        # print(weights)

    plt.figure()
    plotData(data)
    plotWeights = weights
    # print(plotWeights)
    # plotWeights[0][0] = weights[0][1]
    # plotWeights[0][1] = - weights[0][0]
    plt.plot(plotWeights.flatten() * 10)
    plt.title(i)
    plt.show()

