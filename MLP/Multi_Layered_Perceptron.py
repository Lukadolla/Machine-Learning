import numpy as np
import math


class MLP:
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        self.num_inputs = num_inputs  # number of inputs
        self.num_hidden_units = num_hidden_units  # number of hidden units
        self.num_outputs = num_outputs  # number of outputs

        self.input_neurons = [0] * num_inputs  # array to store input neurons
        self.hidden_neurons = [0] * num_hidden_units  # array to store hidden neurons
        self.outputs = [0] * num_outputs  # array to store mlp outputs

        self.activations_lower = [0] * num_hidden_units  # array to store lower activation values
        self.activations_upper = [0] * num_outputs  # array to store upper activation values

    lower_weights = []  # array to store the lower weight values
    upper_weights = []  # array to store the upper weight values
    changes_lower = []  # array to store lower weight changes of size input * hidden_units
    changes_upper = []  # array to store upper weight changes of size hidden_units * output

    # method that fills the changes arrays with 0s
    def build(self):

        for _ in range(self.num_hidden_units):
            temp_arr = []
            for _ in range(self.num_outputs):
                temp_arr.append(0)
            self.changes_upper.append(temp_arr)

        for _ in range(self.num_inputs):
            temp_arr = []
            for _ in range(self.num_hidden_units):
                temp_arr.append(0)
            self.changes_lower.append(temp_arr)

    # method that fills the upper and lower weight arrays with random numbers between 0 and 1.
    def random(self):

        for _ in range(self.num_hidden_units):
            temp_arr = []
            for _ in range(self.num_outputs):
                temp_arr.append(np.random.uniform(0, 1))
            self.upper_weights.append(temp_arr)

        for _ in range(self.num_inputs):
            temp_arr = []
            for _ in range(self.num_hidden_units):
                temp_arr.append(np.random.uniform(0, 1))
            self.lower_weights.append(temp_arr)

    # method that updates the upper and lower weights.
    def update_weights(self, learning_rate):

        for x in range(self.num_hidden_units):
            for y in range(self.num_outputs):
                self.upper_weights[x][y] += self.changes_upper[x][y] * learning_rate
                self.changes_upper[x][y] = 0

        for x in range(self.num_inputs):
            for y in range(self.num_hidden_units):
                self.lower_weights[x][y] += self.changes_lower[x][y] * learning_rate
                self.changes_lower[x][y] = 0

    # computes activations of lower and upper layers using forward propagation.
    # for the XOR question we use sigmoid() and for Sin we use hyperbolic_tangent()
    # the sigmoid boolean determines if the sigmoid() or hyperbolic_tangent() methods will be used
    def forward(self, input_vectors, sigmoid):

        self.input_neurons = input_vectors

        for x in range(self.num_hidden_units):
            act_neuron = 0
            for y in range(self.num_inputs):
                act_neuron += self.input_neurons[y] * self.lower_weights[y][x]

            if sigmoid:
                act_neuron = self.sigmoid(act_neuron, True)
            else:
                act_neuron = self.hyperbolic_tangent(act_neuron, True)

            self.activations_lower[x] = act_neuron
            self.hidden_neurons[x] = act_neuron

        for x in range(self.num_outputs):
            act_output = 0
            for y in range(self.num_hidden_units):
                act_output += self.hidden_neurons[y] * self.upper_weights[y][x]

            if sigmoid:
                act_output = self.sigmoid(act_output, True)
            else:
                act_output = self.hyperbolic_tangent(act_output, True)

            self.activations_upper[x] = act_output
            self.outputs[x] = act_output

        return 0

    # computes error. computes activation derivatives based on XOR or SIN.
    # for the XOR question we use sigmoid() and for Sin we use hyperbolic_tangent()
    # the sigmoid boolean determines if the sigmoid() or hyperbolic_tangent() methods will be used
    # returns computed error
    def backwards(self, target, sigmoid):
        delta_hidden = [0.0] * self.num_hidden_units

        for x in range(self.num_hidden_units):
            error = 0
            for y in range(self.num_outputs):
                if sigmoid:
                    delta = self.sigmoid(self.outputs[y], False) * (target[y] - self.outputs[y])
                else:
                    delta = self.hyperbolic_tangent(self.outputs[y], False) * (target[y] - self.outputs[y])

                error += delta * self.upper_weights[x][y]
                self.changes_upper[x][y] = delta * self.hidden_neurons[x]

            if sigmoid:
                delta_hidden[x] = self.sigmoid(self.hidden_neurons[x], False) * error
            else:
                delta_hidden[x] = self.hyperbolic_tangent(self.hidden_neurons[x], False) * error

        for x in range(self.num_inputs):
            for y in range(self.num_hidden_units):
                self.changes_lower[x][y] = delta_hidden[y] * self.input_neurons[x]

        return np.mean(np.abs(np.subtract(target, self.outputs)))

    # Sigmoid with range 0 to 1
    def sigmoid(self, sig_input, is_forward):
        if is_forward:
            return 1.0 / (1.0 + math.exp(-sig_input))
        else:
            return sig_input * (1.0 - sig_input)

    # Tanh with range -1 to 1
    def hyperbolic_tangent(self, tanh_input, is_forward):
        if is_forward:
            return math.tanh(tanh_input)
        else:
            return 1 - ((math.tanh(tanh_input)) ** 2)
