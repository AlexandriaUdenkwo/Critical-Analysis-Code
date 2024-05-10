#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alexandriaudenkwo
"""

import numpy as np
import matplotlib.pyplot as plt

#Part A
dim = 100

v = []
#make vectors
for i in range(3):
    temp_vec = np.random.normal(loc=0, scale=1, size=dim)
    norm = temp_vec / np.linalg.norm(temp_vec)
    v.append(norm)

print
for i in range(3):
    print("Random Vector " + str(i+1), v[i] )
    
#Part B
print("Scalar product of v1,v2", np.dot(v[0],v[1]))
print("Scalar product of v2,v3", np.dot(v[1],v[2]))
print("Scalar product of v3,v1", np.dot(v[2],v[0]))

print("Scalar product of v1,v1", np.dot(v[0],v[0]))
print("Scalar product of v2,v2", np.dot(v[1],v[1]))
print("Scalar product of v3,v3", np.dot(v[2],v[2]))

#part c
N= 500

#make coefficients

a=np.random.normal(0, np.sqrt(20), size=N)
b=np.random.normal(0, np.sqrt(5), size=N)
c=np.random.normal(0, np.sqrt(0.5), size=N)


x = np.zeros(dim)  
x = a[:,np.newaxis]*v[0] + b[:,np.newaxis]*v[1] + c[:,np.newaxis]*v[2] 


noise = np.random.normal(loc=0, scale=1, size=x.shape)

#part d
x_bob = x+ noise

#part e
plt.figure(1)
plt.scatter(x_bob[:,1], x_bob[:,0])
plt.xlabel('Second Characteristic')
plt.ylabel('First Characteristic')


print("For first characteristic, variance is ", np.var(x_bob[:,0]))
print("For second characteristic, variance is ", np.var(x_bob[:,1]))

#part f
#the dot product u dot x or xucostheta where theta is the angle between the two vectors

#part g
vnew = []
#make vectors
for i in range(2):
    temp_vec = np.random.normal(loc=0, scale=1, size=dim)
    norm = temp_vec / np.linalg.norm(temp_vec)
    norm = norm[:, np.newaxis]
    vnew.append(norm)
#first projection
pro1 = np.dot(x_bob, vnew[0])

#second projection
pro2 = np.dot(x_bob, vnew[1])
print("For first projection, variance is ", np.var(pro1))
print("For second projection, variance is ", np.var(pro2))

plt.figure(2)
plt.scatter(pro2, pro1)
plt.xlabel('Projection onto 2nd random vector')
plt.ylabel('Projection onto 1st random vector')

#part h
cov = np.cov(x_bob, rowvar=False)

# Compute the eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(cov)


order = np.argsort(eigvals)[::-1]
order_eigvecs = eigvecs[:, order]


#part i
#third projection
pro3 = np.dot(x_bob, order_eigvecs[:,0])

#fourth projection
pro4 = np.dot(x_bob, order_eigvecs[:,1])

#fifth projection
pro5 = np.dot(x_bob, order_eigvecs[:,2])

print("For projection onto first eigenvector, variance is ", np.var(pro3))
print("For projection onto second eigenvector, variance is ", np.var(pro4))
print("For projection onto third eigenvector, variance is ", np.var(pro5))

#Part j
print("Scalar product of v1,eig1", np.dot(v[0],order_eigvecs[:,0]))
print("Scalar product of v2,eig2", np.dot(v[1],order_eigvecs[:,1]))
print("Scalar product of v3,eig3", np.dot(v[2],order_eigvecs[:,2]))

plt.figure(3)
plt.scatter(pro4, pro3)
plt.xlabel('Projection onto 2nd eigenvector')
plt.ylabel('Projection onto 1st eigenvector')
#part k
def marchenko_pastur(x, sigma_squared, dim, N):
    r = dim / N
    lambda_plus = sigma_squared*(1 + np.sqrt(r))**2
    lambda_minus = sigma_squared*(1 - np.sqrt(r))**2
    return 1/(2*np.pi*sigma_squared*r*x)*np.sqrt((lambda_plus - x)*(x - lambda_minus))

#histogram
plt.figure(4)
plt.hist(eigvals, bins=100, density=True, label='Eigenvalues')

#Marchenko-Pastur distribution
x = np.linspace(0, 20, 10000)
sigma = 1
plt.plot(x, marchenko_pastur(x, sigma, dim, N), 'r-', label='Marchenko-Pastur distribution')

plt.legend()
plt.xlabel('Eigenvalue')


#part l
plt.figure(5)
x = np.linspace(0, 2, 10000)
sigma = 1
plt.plot(x, marchenko_pastur(x, sigma, dim, 70), 'r-', label='Marchenko-Pastur distribution')
plt.show()

