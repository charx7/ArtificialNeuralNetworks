import numpy as np
from generateRandomData import generateRandomData
from perceptron import classify
from matplotlib import pyplot as plt
from tqdm import tqdm

# create n_d independent datasets wit P points and N dimensions
n_d = 100

# let the perceptron run and check whether it converges
# increase Alpha (alpha =0.75,1.0,1.25,...3.0) and obtain Q (the number of successful runs
# as a function of alpha)

N_values = [5, 20, 100]
# TODO increase the alphas to calculate
alphas = np.linspace(0.5, 3, 40)

# each row is for one P_value (so we do multiple experiments)
# so row 1 will be the y values for P = 20 and the x values are the alpha values
# then we can plot the results x,y, since Q is a function of alpha
# TODO reuse Q to store the generalization error
Q = np.zeros([len(N_values),len(alphas)])

for i, N in enumerate(N_values):
    print('\nComputing the minover of: ', N, ' dimensional space...\n')
    for j, alpha in enumerate(tqdm(alphas)):
        # How many points are we generating
        P = int(np.round(alpha * N))   #
        success_counter = 0
        epochs = 500
        learning_rate = 1.0/N #Number of dimensions
        for trial in range(0, n_d):
            # TODO this should also save and output the teacher perceptron and the labels
            data, weights = generateRandomData(P, N)
            # TODO replace the success counter with the call of the generalization error
            # Implement the function elsewhere
            # the result of classify() is either 0 or 1
            success_counter = success_counter + classify(data,epochs,learning_rate, N)
        Q[i,j] = success_counter/n_d #this describes the probability of sucessful classification for P = i and alpha = j

# TODO Modify labels and pretty plotz
plt.figure()
for i in range(0,len(N_values)):
    currentLabel = str(N_values[i]) + ' dimensions'
    plt.plot(alphas, Q[i], label = currentLabel)
    plt.title("Separation probability based on alpha")
    plt.xlabel("alpha = P/N")
    plt.ylabel("probability of separation")
plt.legend()
plt.show()
