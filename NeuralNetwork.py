from matrix import matrix


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.bias_h = matrix(self.hidden_nodes, 1, randomize=True)
        self.bias_o = matrix(self.output_nodes, 1, randomize=True)

        self.weights_ih = matrix(
            self.hidden_nodes, self.input_nodes, randomize=True)
        self.weights_ho = matrix(
            self.output_nodes, self.hidden_nodes, randomize=True)

    def feedforward(self, inputs):
        self.input = matrix.array_to_matrix(inputs)
        self.hidden = matrix.multiply(self.weights_ih, self.input)
        self.hidden = matrix.add(self.hidden, self.bias_h)
        self.hidden.sigmoid()
        
        self.output = matrix.multiply(self.weights_ho, self.hidden)
        self.output = matrix.add(self.output, self.bias_o)
        self.output.sigmoid()
        
        self.output.show()
