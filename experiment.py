# What I don't understand is how to create a smooth curve
# we evaluate the probabilites only for a few alpha values, right?
# so its not a continuous function. Thats not a problem, is it?
# Also I don't understand how to pick the number of epochs and
# the learning rate. Can we just do what we want?

import numpy as np
from generateRandomData import generateRandomData
from perceptron import classify
from matplotlib import pyplot as plt
from tqdm import tqdm

# Pseudocode:
# pick P (number of Points)
# pick an Alpha, that determines the ratio of P/N
# determine N (number of dimensions) as P/Alpha
# create n_d independent datasets wit P points and N dimensions
n_d = 100
# let the perceptron run and check whether it converges
# increase Alpha (alpha =0.75,1.0,1.25,...3.0) and obtain Q (the number of successful runs
# as a function of alpha)

N_values = [20, 50, 80, 110, 140]
alphas = np.linspace(0.5, 3, 40)

# each row is for one P_value (so we do multiple experiments)
# so row 1 will be the y values for P = 20 and the x values are the alpha values
# then we can plot the results x,y, since Q is a function of alpha
Q = np.zeros([len(N_values),len(alphas)])

for i, N in enumerate(N_values):
    print('\nComputing the l.s. of: ', N, ' dimensional space...\n')
    for j, alpha in enumerate(tqdm(alphas)):
        P = int(np.round(alpha * N))   # int(np.round(P/alpha))
        success_counter = 0
        epochs = 200
        learning_rate = 1.0/N #Number of dimensions
        for trial in range(0, n_d):
            data, weights = generateRandomData(P, N)
            # the result of classify() is either 0 or 1
            success_counter = success_counter + classify(data,epochs,learning_rate, N)
        Q[i,j] = success_counter/n_d #this describes the probability of sucessful classification for P = i and alpha = j

plt.figure()
for i in range(0,len(N_values)):
    currentLabel = str(N_values[i]) + ' dimensions'
    plt.plot(alphas, Q[i], label = currentLabel)
    plt.title("Separation probability based on alpha")
    plt.xlabel("alpha = P/N")
    plt.ylabel("probability of separation")
plt.legend()
plt.show()
