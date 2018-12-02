from matplotlib import pyplot as plt
import numpy as np
from generateRandomData import generateRandomData


data, weights = generateRandomData(200,2)
data[:,2] = np.round(data[:,2])

plt.figure()
my_colors = {'1.0' : 'red','-1.0' : 'green'}
c = [my_colors[str(val)] for val in data[:,2]]
plt.scatter(data[:, 0], data[:, 1], color=c)
plt.show()
