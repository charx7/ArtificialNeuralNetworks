from generateRandomData import generateRandomData
import numpy as np

# X datapoints in Y dimenions
data, weights = generateRandomData(10,2)

epochs = 1 # Number of epochs
learnRate = 1 / 2
for i in range(epochs):
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        if iter == 1:
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:2].reshape(1,2)
            # Assign the weights as the dot product
            weights = np.multiply([[1, 1]], currentPoint)
            print(weights)
            weights = (1/2) * weights
            print(weights)

        print('im on another iteration!')
