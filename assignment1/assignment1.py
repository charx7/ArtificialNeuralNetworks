from generateRandomData import generateRandomData
from testDataGeneration import plotData
import numpy as np
from matplotlib import pyplot as plt

def calculateError(weights, data, dims):
    errors = 0
    numberOfPoints = data.shape[0]
    for iter in range(numberOfPoints):
        currentPoint = data[iter][0:dims].reshape(1, dims)
        currentLabel = data[iter][dims]
        # localPotential = np.dot(weights.flatten(), currentPoint.flatten())
        classification = np.sign(np.dot(weights.flatten(), currentPoint.flatten()))
        print('The classification label is: ',classification)
        print('The correct label is: ', np.sign(currentLabel))
        if classification != np.sign(currentLabel):
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
    data, weights = generateRandomData(20,30)
    # Plot you dataz
    #plotData(data)

    # min and max of generated data
    min = np.amin(data)
    max = np.amax(data)
    # Declare a linspace for the data generated
    x = np.linspace(min, max, 20)

    epochs = 10 # Number of epochs
    # Get the dimensions
    dims = data.shape[1] - 1
    # Set the learn rate according to the dims
    learnRate = 1 / dims

    # Ones vector generation
    onesVector = np.ones(dims)
    weights = np.zeros(dims)
    accuracies = []
    # Get the number of points for the inner loop
    numberOfPoints = data.shape[0]

    for i in range(epochs):
        print("epoch " + str(i))
        for iter in range(numberOfPoints):
            # Get the current point and reshape it to be able to use the dot-product
            currentPoint = data[iter][0:dims].reshape(1, dims)
            currentLabel = data[iter][dims]

            # Local potential Calculation
            localPotential = np.dot(weights.flatten(), currentPoint.flatten()) * currentLabel
            # Check if the dot product is < 0 to update
            if localPotential <= 0:
                weights = weights + (learnRate *  currentPoint) * currentLabel

        # Append accuracies
        accuracies.append(calculateError(weights, data, dims))

    # Print decision boundry for viz
    if dims == 2:
        plt.figure()
        prettyPlot(x, weights)
        plt.show()

    print('The accuracies of our data are: ',accuracies)
    plt.figure()
    plt.plot([i for i in range(0,epochs)], accuracies)
    plt.title("accuracy over time")
    plt.show()

    print('The final weights are: ', weights)
