print('Im working')
import numpy as np

def generateRandomData(dataPoints, dims):
    """
    Returns Randomly generated data form N(0,1)
    and weight vector initialized at 0
    Parameters
    ----------
    dataPoints : int
        How many data points we want to generate `dataPoints`.
    dims: int
        In how many dimensions we want those data points
    """
    # Specify the dimension of the random vector
    # X data points in Y dimensions
    examples = dataPoints
    dimensions = dims
    dimensions = (examples, dimensions)
    # Parameters for the normal generator
    mean = 0
    var = 1
    data = np.random.normal(loc=mean, scale=var, size=dimensions)
    print('The shape of our data vector is: \n', data.shape)

    # Initialize the weight vectors as zeros
    weightVector = np.zeros((dims,1))
    print('The shape of our weights vector is: \n', weightVector.shape)

    # Now for the labels
    s = np.random.uniform(0,1,examples)
    print('The shape of our uniform vector is: \n', s.shape)

    # Initialize an empty numpy array
    labels = np.array([])
    # Generate labels -1 or 1
    for i in range(s.shape[0]):
        if s[i] < 0.5:
            labels = np.hstack((labels, 1))
        else:
            labels = np.hstack((labels, -1))

    print('The shape of our labels is: \n', labels.shape)
    verticalLabels = labels.reshape((examples, 1))
    print('The shape of our labels to attach is: \n', verticalLabels.shape)

    # Attach the labels to the randomly generated dataz
    labeledData = np.hstack((data, verticalLabels))
    print('The shape of our labeled data is: \n', labeledData.shape)

    return labeledData, weightVector

