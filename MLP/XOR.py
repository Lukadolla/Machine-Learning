import numpy as np
from Multi_Layered_Perceptron import MLP

# defining variables for Multi Layered Perceptron
EPOCHS = 10000
LEARNING_RATE = 0.2
INPUTS = 2
HIDDEN = 4
OUTPUTS = 1

# initializing Multi Layered Perceptron
mlp = MLP(INPUTS, HIDDEN, OUTPUTS)
mlp.build()
mlp.random()

# defining inputs and desired outputs
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_output_desired = [[0], [1], [1], [0]]

# creating file with the learning rate and hidden unit count in the name
# all output is saved there
f = open("XOR_" + str(LEARNING_RATE) + "_" + str(HIDDEN) + ".txt", "w")

f.write("XOR with a Learning Rate of: " + str(LEARNING_RATE) + " and Hidden Unit count of: " + str(HIDDEN) + "\n")
f.write("\n===PRE-TRAINING TESTING===\n")

# outputting the Multi Layered Perceptron details
f.write('\nMulti Layered Perceptron Size: ' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS) + '\n')
f.write('Epochs: ' + str(EPOCHS) + '\n')
f.write('Learning Rate: ' + str(LEARNING_RATE) + '\n\n')

print("THE PROGRAM IS RUNNING, ALL OUTPUT IS PRINTED TO FILE")

# running pre-training testing on inputs using forward() and sigmoid()
for x in range(0, len(xor_inputs)):
    mlp.forward(xor_inputs[x], True)
    f.write("Target: " + str(xor_output_desired[x]) + "\tOutput: " + str(mlp.outputs) + "\n")

# running training on the inputs using forward() and sigmoid() and calculating error using backward()
# weight updating occurs every epoch
f.write("\n===TRAINING===\n")

for x in range(0, EPOCHS):
    error = []
    for y in range(0, len(xor_inputs)):
        mlp.forward(xor_inputs[y], True)
        error.append(mlp.backwards(xor_output_desired[y], True))
        mlp.update_weights(LEARNING_RATE)

    if x % 200 == 0:
        f.write("\nEpoch: " + str(x) + "\tError for Epoch: " + str(np.mean(error)))

# running testing on the inputs using forward() and sigmoid() and displaying the results
f.write("\n\n===TESTING===\n\n")

avg_diff = []
for x in range(0, len(xor_inputs)):
    mlp.forward(xor_inputs[x], True)
    avg_diff.append(np.abs(np.subtract(xor_output_desired[x], mlp.outputs)))
    f.write("Target: " + str(xor_output_desired[x]) + "\tOutput: " + str(mlp.outputs) + "\n")
    f.write("Difference: " + str(np.abs(np.subtract(xor_output_desired[x], mlp.outputs))) + "\n\n")
f.write("Average difference: " + str(np.mean(avg_diff)))
print("\nDONE!")
