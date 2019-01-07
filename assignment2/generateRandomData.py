print('Im working')
import numpy as np



def create_labels(teacher, data):
    # Use the teacher perceptron to generate the data labels
    labels = np.sign(np.dot(teacher, data))
    return labels

# TODO change the function to also give us the teacher + correct labels
def generateRandomData(dataPoints, dims, teacher):
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
    # print('The shape of our data vector is: \n', data)

    # Initialize the weight vectors as zeros
    weightVector = np.zeros((dims,1))
    #print('The shape of our weights vector is: \n', weightVector.shape)

    # Now for the labels
    s = np.random.uniform(0,1,examples)
    #print('The shape of our uniform vector is: \n', s.shape)

    # Initialize an empty numpy array
    labels = np.array([])

    # Generate labels -1 or 1 based on our teacher peceptron
    for i in range(dataPoints):
        labels = np.hstack((labels, create_labels(teacher, data[i,:])))

    #print('The shape of our labels is: \n', labels.shape)
    verticalLabels = labels.reshape((examples, 1))
    #print('The shape of our labels to attach is: \n', verticalLabels.shape)

    # Attach the labels to the randomly generated dataz
    labeledData = np.hstack((data, verticalLabels))
    #print('The shape of our labeled data is: \n', labeledData.shape)

    return labeledData, weightVector

# Function that generates a random teacher vector
def generateRandomTeacher(dimensions):
    print('The norm of the points is: ', np.linalg.norm(np.random.randn(dimensions)))
    random_weights = np.random.randn(dimensions)
    return np.sign(random_weights), random_weights

# Chunk code execution for tests
def main():
    from minonver import classify
    print('Im on main hi!')
    # To be replaced by N
    dimensions = 4
    dataPoints = 20

    # Create the randomly generated teacher perceptron
    teacher, teacher_weights = generateRandomTeacher(dimensions)
    print('The norm squared of the teacher is: ', (np.linalg.norm(teacher))**2)
    print('\nThe Teacher weights are: \n', teacher_weights)
    # Generate random data points
    data, weights = generateRandomData(20, 4, teacher)
    print('The random data is: \n', data)
    print('The random weights are: \n',weights)

    # Run the minover algo
    epochs = 10 # Number of epochs
    learning_rate = 1.0 / dimensions #Number of dimensions
    N = dimensions
    success_counter = 0 # To be removed
    success_counter = success_counter + classify(data,epochs,learning_rate, N)

# Execute main function if called from the terminal
if __name__ == '__main__':
    main()
