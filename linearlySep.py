import numpy as np
import math
from generateRandomData import generateRandomData
from perceptron import classify
from matplotlib import pyplot as plt

# Number of random data sets that will be generated
n_d = 50

dimension = 20 # Right now we are doing it for 20
alphas = np.linspace(0.5, 4, 15)

for alpha in alphas:
    # Number of points to be generated according to alpha
    N = math.ceil(alpha*dimension)
    print('The number of points generated will be: ', N)
    # Number of succeses
    success_counter = 0
    epochs = 5
    learning_rate = 1 / dimension

    for trial in range(1, n_d):
        # Generate random data
        data, weights = generateRandomData(N, dimension)
        # Do the rosenblats
        success_counter = success_counter + classify(data,epochs,learning_rate, dimension)

    print(success_counter)
