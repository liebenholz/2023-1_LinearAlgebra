# A scalar-valued function of n vectors
import numpy as np 
def func(x):
    res = x[0] + x[1] - x[3]**2
    return res

x = np.array([-1, 0, 1, 10])
func(x)
# func(x) = -1 + 0 - 10^2 = -101

func = lambda x: x[0] + x[1] - x[3]**2
x = np.array([-1, 0, 1, 10])
func(x)
# func(x) = -1 + 0 - 10^2 = -101

# Superposition equality
a = np.random.random(3)
x = np.random.random(3)
y = np.random.random(3)

def func(a, x, y):
    alpha = np.random.uniform()
    beta = np.random.uniform()
    return np.inner(a, alpha*x + beta*y) == alpha*np.inner(a, x) + beta*np.inner(a, y)

func(a, x, y)

# Let’s define the average function in Python and check its value of a specific vector
func = lambda x: sum(x)/len(x)
v1 = [1, 2, 3, 4, 5]
v2 = [-1, 0, 1]
print(func(v1))
print(func(v2))

# Numpy also contains an average function, which can be called with np.mean
print(np.mean(v1))
print(np.mean(v2))

# Regression Model
import matplotlib.pyplot as plt
import import_ipynb
from houseSalesData import house_sales_data
D =  house_sales_data()
price = D['orice']
area = D['area']
V = 54.4017
beta = np.array ( [147.7251, -18.8534])
predicted = v + beta[0]*area + beta[1]*beds
plt.scatter(price, predicted)
plt.plot((0,800),(0,800), ls='--', x = 'r')
plt.ylim(0,800) 
plt.xlim(0,800)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")