# What I don't understand is how to create a smooth curve
# we evaluate the probabilites only for a few alpha values, right?
# so its not a continuous function. Thats not a problem, is it?
# Also I don't understand how to pick the number of epochs and
# the learning rate. Can we just do what we want?

import numpy as np
from generateRandomData import generateRandomData
from perceptron import classify
from matplotlib import pyplot as plt

# Pseudocode:
# pick P (number of Points)
# pick an Alpha, that determines the ratio of P/N
# determine N (number of dimensions) as P/Alpha
# create n_d independent datasets wit P points and N dimensions
n_d = 50
# let the perceptron run and check whether it converges
# increase Alpha (α=0.75,1.0,1.25,...3.0) and obtain Q (the number of successful runs
# as a function of alpha)

P_values = [20] #, 30, 50, 70, 100]
alphas = np.linspace(0.5, 4, 15)

# each row is for one P_value (so we do multiple experiments)
# so row 1 will be the y values for P = 20 and the x values are the alpha values
# then we can plot the results x,y, since Q is a function of alpha
Q = np.zeros([5,15])

for i, P in enumerate(P_values):
    for j, alpha in enumerate(alphas):
        print(j)
        N = int(np.round(P/alpha))
        success_counter = 0
        epochs = 5
        learning_rate = 0.5
        for trial in range(1, n_d):
            data, weights = generateRandomData(P, N)
            # the result of classify() is either 0 or 1
            success_counter = success_counter + classify(data,epochs,learning_rate)

        Q[i,j] = success_counter/n_d #this describes the probability of sucessful classification for P = i and alpha = j

plt.figure()
plt.plot(alphas, Q[1])
plt.title("Separation probability based on α")
plt.xlabel("α = P/N")
plt.ylabel("probability of separation")
plt.show()
