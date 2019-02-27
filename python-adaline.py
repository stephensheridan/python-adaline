"""
Title: Very simple ADALINE network
Author: Stephen Sheridan (ITB) https://github.com/stephensheridan
Date: 09/03/2017
"""

import numpy as np
import matplotlib.pyplot as plt
import math

LEARNING_RATE = 0.45

# Step function
def step(x):
    if (x > 0):
        return 1
    else:
        return -1;
    
"""
You can comment out either the first or second problem to see how the ADALINE network performs with
linearly separable and non linearly separable problems.
"""

# F I R S T   P R O B L E M - L O G I C A L   O R   L I N E A R
# input dataset representing the logical OR operator (including constant BIAS input of 1)
INPUTS = np.array([[-1,-1,1],
                   [-1,1,1],
                   [1,-1,1],
                   [1,1,1] ])
# output dataset - Only output a -1 if both inputs are -1          
OUTPUTS = np.array([[-1,1,1,1]]).T


# S E C O N D   P R O B L E M - L O G I C A L   X O R - N O N   L I N E A R
# input dataset representing the logical OR operator (including constant BIAS input of 1)
#INPUTS = np.array([[-1,-1,1],
#                   [-1,1,1],
#                   [1,-1,1],
#                   [1,1,1] ])
# output dataset - Only output a -1 if both inputs are -1          
#OUTPUTS = np.array([[-1,1,1,-1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice for testing)
np.random.seed(1)

# initialize weights randomly with mean 0
WEIGHTS = 2*np.random.random((3,1)) - 1
print "Random Weights before training", WEIGHTS

# Use this list to store the errors
errors=[]

# Training loop
for iter in xrange(100):

    for input_item,desired in zip(INPUTS, OUTPUTS):
        
        # Feed this input forward and calculate the ADALINE output
        ADALINE_OUTPUT = (input_item[0]*WEIGHTS[0]) + (input_item[1]*WEIGHTS[1]) + (input_item[2]*WEIGHTS[2])

        # Run ADALINE_OUTPUT through the step function
        #DALINE_OUTPUT = step(ADALINE_OUTPUT)

        # Calculate the ERROR generated
        # Squared error should be use as described in fausset book.
        ERROR = .5*(desired - ADALINE_OUTPUT) ** 2
        
        
        # Store the ERROR
        errors.append(ERROR)
        
        # Update the weights based on the delta rule
        
        
        WEIGHTS[0] +=  LEARNING_RATE * (desired - ADALINE_OUTPUT) * input_item[0]
        WEIGHTS[1 +=  LEARNING_RATE * (desired - ADALINE_OUTPUT) * input_item[1]
        WEIGHTS[2] +=  LEARNING_RATE * (desired - ADALINE_OUTPUT) * input_item[2]


print "New Weights after training", WEIGHTS
for input_item,desired in zip(INPUTS, OUTPUTS):
    # Feed this input forward and calculate the ADALINE output
    ADALINE_OUTPUT = (input_item[0]*WEIGHTS[0]) + (input_item[1]*WEIGHTS[1]) + (input_item[2]*WEIGHTS[2])

    # Run ADALINE_OUTPUT through the step function
    ADALINE_OUTPUT = step(ADALINE_OUTPUT)

    print "Actual ", ADALINE_OUTPUT, "Desired ", desired


# Plot the errors to see how we did during training
ax = plt.subplot(111)
ax.plot(errors, c='#aaaaff', label='Training Errors')
ax.set_xscale("log")
plt.title("ADALINE Errors (2,-2)")
plt.legend()
plt.xlabel('Error')
plt.ylabel('Value')
plt.show()


