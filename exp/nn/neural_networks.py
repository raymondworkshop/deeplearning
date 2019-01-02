# coding: utf-8

"""
The core foucs of the repository
      28-12-2018 creation By Raymond
"""

import sys
import os

import numpy as np
import functions

# y <- z = x * W
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        #y = softmax(z)
        y = functions.relu(z)
        loss = functions.cross_entropy_error(y, t)

        return loss



def main():

    return 0

if __name__ == "__main__":
    main()