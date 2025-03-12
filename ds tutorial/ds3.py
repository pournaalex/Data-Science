import numpy as np #importing numpy
from scipy.linalg import eig #importing eig
class1= np.array([[2, 3], [3, 4], [4, 5]]) #example 1
class2= np.array([[6, 7], [7, 8], [8, 9]]) #example 2
mean1= np.mean(class1, axis=0) #mean of example 1 
mean2= np.mean(class2, axis=0) #mean of example 2
S1= np.cov(class1, rowvar=False) #scatter matrix 1
S2= np.cov(class2, rowvar=False) #scatter matrix 2
SW= S1 + S2 #within class scatter matrix
mean_diff= np.array([[1], [2], [3]]) #substracting scatter matrices
SB= np.dot(mean_diff, mean_diff.T)
#finding lda using standard library functions
inverse= np.dot(np.linalg.inv(SW), SB)
eigenvalues, eigenvectors= eig(inverse)
#finding eigen vectors
man_eigenvalues= np.linalg.eigvals(inverse)
#output
print("Eigenvalues using SciPy:", eigenvalues) 
print("Eigenvectors using SciPy:\n", eigenvectors)
print("Eigenvalues using manual computation:", man_eigenvalues)
