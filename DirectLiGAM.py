import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)


from numpy import genfromtxt
my_data = genfromtxt('Datasets/BMI-M.csv', delimiter=',')

my_data = my_data[0:5]
model = lingam.DirectLiNGAM()
model.fit(my_data)

print(model.causal_order_)
print(model.adjacency_matrix_)
