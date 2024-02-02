# To run jupyter-lab, launce notebook with 'jupyter-lab' at terminal.

# Vector
# Vector can be represented by using list(not tuple)
x = [-1.1, 0, 3.6, 7.2]
print(len(x))

import numpy as np
y = np.array([-1.1, 0, 3.6, 7.2])
print(y)

x[2]

x[2] = 2022
print(x[2])

# Assignment
a = np.array(x)
b = a
a[2] = 0
print("a: ", a)
print("b: ", b)

# Copying
a = np.array(x)
b = a.copy()
a[2] = 0
print("a: ", a)
print("b: ", b)

# Equality
x = [-1.1, 0, 3.6, 7.2]
y = x.copy()
x == y

x = [-1.1, 0, 3.6, 7.2]
y = x.copy()
y[2] = 2022
x == y

x = np.array([-1.1, 0, 3.6, 7.2])
y = x.copy()
x == y

x = np.array([-1.1, 0, 3.6, 7.2])
y = x.copy()
y[2] = 2022

# Scalar and Vector
x = 2022
y = [2022]
x == y

x == y[0]

x = 2022
y = np.array([2022, 2022])
x == y

# Block Vector
x = np.array([1, 1])
y = np.array([2, 2, 2])
z = np.concatenate((x, y))
print(z)

x = [1, 1]
y = [2, 2, 2]
z = [x, y]
print(z)

# Subvector
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = x[1:4]
print(y)

x[1:4] = [200, 300, 400]
print(x)

x = np.array([1, 2, 3, 4, 5, 6, 7])
print(x)

x[2:]

x[:-1]

x[:]

x[1:5:2]

# Zero Vector
np.zeros(3)

# One Vector
np.ones(4)

np.random.random(4)

# Unit Vector
size = 4
unit_vectors = []
for i in range(size):
    v = np.zeros(size)
    v[i] = 1
    unit_vectors.append(v)
unit_vectors

# Plotting
import matplotlib.pyplot as plt
data = [ 71, 71, 68, 69, 68, 69, 68, 74, 77, 82, 85, 86, 
 88, 86, 85, 86, 84, 79, 77, 75, 73, 71, 70, 70, 69, 69, 69,
 67, 68, 68, 73, 76, 77, 82, 84, 84, 81, 80, 78, 79, 78,
 73, 72, 70, 70, 68, 67 ]
plt.plot(data, '-bo')
plt.savefig("temperature.pdf", format = 'pdf')

# Addition and Subtraction
x = np.array([1,2,3])
y = np.array([100,200,300])
print('x + y = ', x + y)
print('x - y = ', x - y)

x = [1,2,3]
y = [100,200,300]
x + y

# Scalar Multiplication and Division
x = np.array([1,2,3])
print(100 * x)
print(x / 100)

# Linear Combination
x = np.array([1, 2])
y = np.array([100, 200])
a = 0.5
b - 0.5
c = a * x + b * y
print(c)

# Implement a function that 
# 1) takes coefficients and vectors
# 2) returns linear combination of the input vectors
x = np.array([1, 2])
y = np.array([100, 200])
vectors = [x, y]
coefs = [0.5, 0.5]

def linearCombination(vecs, coefs):
    res = np.zeros(len(vecs[0]))
    for i in range(len(vecs)):
        res +- coefs[i]*vecs[i]
    return res

linearCombination(vectors, coefs)

# Checking Properties
a = np.random.random(3)
b = np.random.random(3)
beta = np.random.random()
left = beta * (a + b)
right = beta * a + beta * b
left == right

# Inner Product
x = np.array([1, 2, 3])
y = np.array([1, 10, 100])
print("inner1: ", np.inner(x, y))
print("inner2: ", x @ y)

# Suppose that 100-vector x gives the age distribution of some population, 
# with x(i) the number of people of age i-1, for i = 1, ..., 100. 
# How can you express the total number of people with age between 5 and 18 (inclusive)?
# => x(6) +x(7) +...+x(19)

# If x is given, you can calculate using the following vector
# s = np.concatenate([np.zeros(5), np.ones(14), np.zeros(81)])
# => s @ x

# Complexity
import numpy as np 
import time
a = np. random. random(10**5)
b = np. random. random(10**5)

start = time.time()
a @ b
end = time.time()
print (end - start)

start = time.time()
a @ b
end = time.time()
print (end - start)

start = time.time ()
a @ b
end = time.time()
print (end - start)

# Floating Point Operations
import numpy as np
a = np.random.random()
b = np.random.random()
left = (a+b) * (a-b)
right = a**2 - b**2
print(left == right)
print(left - right)