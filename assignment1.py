from generateRandomData import generateRandomData
from testDataGeneration import plotData
import numpy as np
from matplotlib import pyplot as plt
# X datapoints in Y dimenions
data, weights = generateRandomData(10,2)
epochs = 10 # Number of epochs
learnRate = 1 / 2

def calculateError(weights, data):
    errors = 0
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        currentPoint = data[iter][0:2].reshape(1, 2)
        currentLabel = data[iter][2]
        localPotential = np.dot(weights.flatten(), currentPoint.flatten())
        if np.sign(localPotential)!=np.sign(currentLabel):
            errors = errors + 1

    return 1-errors/numberOfPoints

def prettyPlot(x, weights):
    # Declare a figure
    fig = plt.figure()
    # Plot the scatter
    my_colors = {'1.0' : 'red','-1.0' : 'green'}
    c = [my_colors[str(val)] for val in data[:,2]]
    plt.scatter(data[:, 0], data[:, 1], color=c)

    # Draw the classifier line
    ax = plt.axes()
    classifier = lambda x: -(x * weights[0][0]) / weights[0][1]
    vfunc = np.vectorize(classifier)
    y = vfunc(x)

    # Display plot :D
    ax.plot(x, y);

if __name__ == '__main__':
    # X datapoints in Y dimenions
    data, weights = generateRandomData(10,2)
    # Plot you dataz
    #plotData(data)

    # min and max of generated data
    min = np.amin(data)
    max = np.amax(data)
    # Declare a linspace for the data generated
    x = np.linspace(min, max, 20)

    epochs = 2 # Number of epochs
    # Get the dimensions
    dims = data.shape[1] - 1
    # Set the learn rate according to the dims
    learnRate = 1 / dims

    # Ones vector generation
    onesVector = np.ones(dims)
    accuracies = []
    for i in range(epochs):
        # Get the number of points for the inner loop
        numberOfPoints = data.shape[0]

        print("epoch " + str(i))
        for iter in range(numberOfPoints):
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:2].reshape(1, 2)
            print(currentPoint)
            currentLabel = data[iter][2]

            if iter == 0 and i == 0:
                # initially assign the weights as the first point
                weights = (np.multiply(onesVector, currentPoint) * currentLabel) * learnRate
            else:
                # Assign the weights as the dot product
                localPotential = np.multiply(np.dot(weights.flatten(), currentPoint.flatten()), currentLabel)
                # Since the cutoff point is zero we check whether the local potential is smaller or equal than 0
                # only then the hebbian term becomes 1 and the weights are updated
                if localPotential > 0:
                    weights = weights # hebbian = 0
                else:
                    weights = np.add(weights, np.multiply(learnRate * currentPoint, currentLabel))
                print('Current weights are:\n ',weights)
                print('end')
            #prettyPlot(x, weights)
        # Print decision boundry for viz
        #if dims == 2:
        #    prettyPlot(x, weights)

        # Append accuracies
        accuracies.append(calculateError(weights, data))

    print(accuracies)
    plt.figure()
    plt.plot([i for i in range(0,epochs)], accuracies)
    plt.title("accuracy over time")
    plt.show()

    print('The final weights are: ', weights)
