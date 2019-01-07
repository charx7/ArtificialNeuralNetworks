import numpy as np
from generateRandomData import generateRandomData
from minover import classify
from matplotlib import pyplot as plt
from tqdm import tqdm

# create n_d independent datasets wit P points and N dimensions
n_d = 100

# let the perceptron run and check whether it converges
# increase Alpha (alpha =0.75,1.0,1.25,...3.0) and obtain Q (the number of successful runs
# as a function of alpha)

N_values = [6, 20, 100]

alphas = np.linspace(0.1, 5, 40)

# each row is for one P_value (so we do multiple experiments)
# so row 1 will be the y values for P = 20 and the x values are the alpha values
# then we can plot the results x,y, since Q is a function of alpha
Q = np.zeros([len(N_values),len(alphas)])

def calculateError(weightsStudent, weightsTeacher):
    nominator = np.dot(weightsStudent, weightsTeacher)
    denominator = np.dot(np.linalg.norm(weightsStudent), np.linalg.norm(weightsTeacher))
    error = 1/np.pi * np.arccos(nominator / denominator)
    return error

def generateRandomTeacher(dimensions):
    # print('The norm of the points is: ', np.linalg.norm(np.random.randn(dimensions)))
    random_weights = np.random.randn(dimensions)
    return np.sign(random_weights), random_weights

for i, N in enumerate(N_values):
    print('\nComputing the minover of: ', N, ' dimensional space...\n')
    for j, alpha in enumerate(tqdm(alphas)):
        # How many points are we generating
        P = int(np.round(alpha * N))   #
        error_list = []
        epochs = 500
        learning_rate = 1.0/N #Number of dimensions
        for trial in range(0, n_d):
            labels, teacher = generateRandomTeacher(N)
            data, weights = generateRandomData(P, N, teacher)
            # the minover.classify function returns the weights after either convergence or max number of epochs
            studentWeights = classify(data,epochs,learning_rate, N)
            # calculate the angle between the studentVector and the teacherVector to get the generalization error
            error_list.append(calculateError(studentWeights, teacher))
        Q[i,j] = np.mean(error_list) #this describes the probability of sucessful classification for P = i and alpha = j

# TODO Modify labels and pretty plotz
plt.figure()
for i in range(0,len(N_values)):
    currentLabel = str(N_values[i]) + ' dimensions'
    plt.plot(alphas, Q[i], label = currentLabel)
    plt.title("Generalization error based on alpha")
    plt.xlabel("alpha = P/N")
    plt.ylabel("Generalization error")
plt.legend()
plt.show()
