import numpy as np
import math
from Multi_Layered_Perceptron import MLP

# defining variables for Multi Layered Perceptron
EPOCHS = 10000
LEARNING_RATE = 0.01
INPUTS = 4
HIDDEN = 6
OUTPUTS = 1

# initializing Multi Layered Perceptron
mlp = MLP(INPUTS, HIDDEN, OUTPUTS)
mlp.build()
mlp.random()

# defining input and desired output arrays as well as arrays to store the test input and desired test output
sin_inputs = []
sin_desired_output = []
sin_test = []
sin_test_out = []

# filling the sin_input array with vectors of 4 random numbers from -1 to 1.
# the sin_input array is filled with 400 and then the rest is stored in the sin_test array to be used as a
# comparison later.
for x in range(0, 500):
    temp = []
    for y in range(4):
        temp.append(np.random.uniform(-1, 1))

    if x < 400:
        sin_inputs.append(temp)
    else:
        sin_test.append(temp)

# the sin_input array is manipulated using the formula: sin(x1-x2+x3-x4) to obtain a result the sin of which is then
# stored in the sin_desired_output array.
# similarly, we compute the same value for the 100 sin_test values and store the sin of the result in sin_test_out.
for y in range(0, 400):
    temp = sin_inputs[y][0] - (sin_inputs[y][1] + sin_inputs[y][2]) - sin_inputs[y][3]
    sin_desired_output.append([math.sin(temp)])

    if y < 100:
        temp = sin_test[y][0] - (sin_test[y][1] + sin_test[y][2]) - sin_test[y][3]
        sin_test_out.append([math.sin(temp)])

# creating file with the learning rate and hidden unit count in the name
# all output is saved there
f = open("SIN_" + str(LEARNING_RATE) + "_" + str(HIDDEN) + ".txt", "w")

f.write("SIN with a Learning Rate of: " + str(LEARNING_RATE) + " and Hidden Unit count of: " + str(HIDDEN) + "\n")
f.write("\n===PRE-TRAINING TESTING===\n")

# outputting the Multi Layered Perceptron details
f.write('Multi Layered Perceptron Size: ' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS) + '\n')
f.write('Epochs: ' + str(EPOCHS) + '\n')
f.write('Learning Rate: ' + str(LEARNING_RATE) + '\n\n')

print("THE PROGRAM IS RUNNING, ALL OUTPUT IS PRINTED TO FILE")

# running pre-training testing on inputs using forward() and hyperbolic_tangent()
for x in range(len(sin_inputs)):
    mlp.forward(sin_inputs[x], False)
    f.write("Target: " + str(sin_desired_output[x]) + "\tOutput: " + str(mlp.outputs) + "\n")

print("\nTHIS PROGRAM CAN TAKE AROUND 3MINS TO RUN")
# running training on the inputs using forward() and hyperbolic_tangent() and calculating error using backward()
# weight updating occurs every epoch
f.write("\n===TRAINING===\n")

for x in range(0, EPOCHS):
    error = []
    for y in range(0, len(sin_inputs)):
        mlp.forward(sin_inputs[y], False)
        error.append(mlp.backwards(sin_desired_output[y], False))
        mlp.update_weights(LEARNING_RATE)

    if x % 200 == 0:
        print("\nWE ARE ON EPOCH: " + str(x) + " OUT OF " + str(EPOCHS))
        f.write("\nEpoch: " + str(x) + "\tError for Epoch: " + str(np.mean(error)))

# running testing on the sin_inputs using forward() and hyperbolic_tangent() and displaying the results
f.write("\n\n===TESTING ON SIN_INPUTS===\n\n")

avg_diff_sin_input = []

for x in range(len(sin_inputs)):
    mlp.forward(sin_inputs[x], False)
    avg_diff_sin_input.append(np.abs(np.subtract(sin_desired_output[x], mlp.outputs)))
    # printing every 2nd result to make file more readable
    if x % 2 == 0:
        f.write("Target: " + str(sin_desired_output[x]) + "\tOutput: " + str(mlp.outputs) + "\n")
        f.write("Difference: " + str(np.abs(np.subtract(sin_desired_output[x], mlp.outputs))) + "\n\n")

# running testing on the sin_test using forward() and hyperbolic_tangent() and displaying the results
f.write("\n\n===TESTING ON SIN_TEST===\n\n")

avg_diff_sin_test = []

for x in range(len(sin_test)):
    mlp.forward(sin_test[x], False)
    avg_diff_sin_test.append(np.abs(np.subtract(sin_test_out[x], mlp.outputs)))
    # printing every 2nd result to make file more readable
    if x % 2 == 0:
        f.write("Target: " + str(sin_test_out[x]) + "\tOutput: " + str(mlp.outputs) + "\n")
        f.write("Difference: " + str(np.abs(np.subtract(sin_test_out[x], mlp.outputs))) + "\n\n")

f.write("Average Difference for sin_input: " + str(np.mean(avg_diff_sin_input)))
f.write("\nAverage Difference for sin_test: " + str(np.mean(avg_diff_sin_test)))
print("\nDONE!")
