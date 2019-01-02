# coding: utf-8

"""
The common functions
"""

import numpy as np

def softmax(x):
    return 0

def relu(x):
    return np.maximum(0, x)

# loss functions
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def main():
    #t = 
    
    return 0

"""
def __name__ == "__main__":
    main()
"""