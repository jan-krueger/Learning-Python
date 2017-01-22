from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the reandom number generator, so it generates the same numbers every
        # time the program runs
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connetion.
        # We assign random weights to a 3x1 matrix, with values in [-1, 1] range
        # and mean 0.
        self.synatpic_weights = 2 * random.random((3, 1)) - 1

    # The sigmoid function, which describes an s shaped curve. We pass the weighted
    # sum of the inputs through this functiuon to normalise them between 0 and 1.
    def __sigmoid(self, x):
        return (1 / (1 + exp(-x)))

    # Gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return (x * (1 - x))


    def train(self, training_set_inputs, training_set_outputs, iterations):
        for i in range(iterations):
            # pass the training set through our neural network
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            # multiply the error by the input and again by the gradient of the
            # sigmoid curve
            adjustments = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust the weights
            self.synatpic_weights += adjustments

    def predict(self, inputs):
        # pass inputs through or neural network (our single neuron)
        return self.__sigmoid(dot(inputs, self.synatpic_weights))

if __name__ == '__main__':

    #initalise a single neuron neural network
    neural_network = NeuralNetwork()

    print ('Random starting synatpic weights:')
    print (neural_network.synatpic_weights)

    # The training set. We have 4 examples, each consiting of 3 input values
    # and 1 output value
    training_set_inputs = array(
        [
            [8, 7, 6],
            [5, 6, 7],
            [10, 11, 12],
            [13, 12, 11]
        ]
    )
    training_set_outputs = array(
        [
            [5, 8, 13, 10]
        ]
    ).T

    #train the neural network using training set
    # Dit it 10000 times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synatpic weihts after training:')
    print(neural_network.synatpic_weights)

    #Test the neural network
    print ('Predicting:')
    print(neural_network.predict(array([8, 7, 6])))
