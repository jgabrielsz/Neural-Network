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
        
        self.learning_rate = 0.2

    def train(self, inputs, targets):
        # sgm(inputs * weights + bias) = hidden
        self.input = matrix.array_to_matrix(inputs)
        self.hidden = matrix.multiply(self.weights_ih, self.input)
        self.hidden = matrix.add(self.hidden, self.bias_h)
        self.hidden.sigmoid()
        
        # sgm(hidden * weights + bias) = output
        self.output = matrix.multiply(self.weights_ho, self.hidden)
        self.output = matrix.add(self.output, self.bias_o)
        self.output.sigmoid()
        
        # error = target - output
        # output --> hidden
        self.expected = matrix.array_to_matrix(targets)
        self.output_error = matrix.subtract(self.expected, self.output)
        
        self.d_output = matrix.d_sigmoid(self.output)
        self.hidden_t = matrix.transpose(self.hidden)

        self.gradient_o = matrix.hadamard(self.output_error, self.d_output)
        self.gradient_o = matrix.multiply_escalar(self.gradient_o, self.learning_rate)
        
        # adjust bias h --> o
        self.bias_o = matrix.add(self.bias_o, self.gradient_o)

        # adjust weights h --> o
        self.weights_ho_deltas = matrix.multiply(self.gradient_o, self.hidden_t)
        self.weights_ho = matrix.add(self.weights_ho, self.weights_ho_deltas)
        
        # hidden --> input
        self.weights_ho_t = matrix.transpose(self.weights_ho)
        self.hidden_error = matrix.multiply(self.weights_ho_t, self.output_error)
        self.d_hidden = matrix.d_sigmoid(self.hidden)
        
        self.input_t = matrix.transpose(self.input)
        self.gradient_h = matrix.hadamard(self.hidden_error, self.d_hidden)
        self.gradient_h = matrix.multiply_escalar(self.gradient_h, self.learning_rate)

        # adjust bias i --> h
        self.bias_h = matrix.add(self.bias_h, self.gradient_h)

        # adjust weights h --> o
        self.weights_ih_deltas = matrix.multiply(self.gradient_h, self.input_t)
        self.weights_ih = matrix.add(self.weights_ih, self.weights_ih_deltas)

    def predict(self, inputs):
        # f(inputs * weights + bias) = hidden
        self.input = matrix.array_to_matrix(inputs)
        self.hidden = matrix.multiply(self.weights_ih, self.input)
        self.hidden = matrix.add(self.hidden, self.bias_h)
        self.hidden.sigmoid()
        
        # f(hidden * weights + bias) = output
        self.output = matrix.multiply(self.weights_ho, self.hidden)
        self.output = matrix.add(self.output, self.bias_o)
        self.output.sigmoid()
        return self.output.matrix[0][0]