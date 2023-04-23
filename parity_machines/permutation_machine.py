import math

import numpy as np

from parity_machines import ParityMachine


class PermutationMachine(ParityMachine):
    def __init__(self, k, n, g=None, seed=None):
        """
        Permutation parity machine implementation. Weights are randomly
        initialized to 0 or 1. The hidden layer is a single neuron with
        `n` inputs.

        :param k: Number of neurons in the hidden layer
        :param n: Number of inputs to each neuron in the hidden layer
        :param g: State vector size
        """

        super().__init__(k, n, seed=seed)

        if g is None:
            self.g = k * n
        else:
            self.g = g

        assert self.g >= k * n, "State vector size must be greater than or equal to the number of weights"

        self.s = np.random.randint(0, 2, self.g)
        self.s_idx = 0
        self.s_filled = False
        self.counter = 0

    def __call__(self, x):
        """
        The output of the hidden layer is the sum of XOR products of
        the input and the weights. The output of the hidden layer is
        then fed into a threshold function which outputs 1 if the sum
        is greater than `n / 2` and 0 otherwise. The output of the
        network is the product of the outputs of the hidden layers.

        :param x:
        :return: Output of the parity machine
        """

        W = np.random.permutation(self.s).reshape(self.k, self.n)
        X = x.reshape(self.k, self.n)

        h = np.bitwise_xor(X, W).sum(axis=1)
        sigma = np.where(h > self.n / 2, 1, 0)
        tau = np.logical_xor.reduce(sigma)

        self.tau = tau
        return tau

    def update(self, tau):
        """

        :param tau:
        :return:
        """

        if tau == self.tau:
            self.s_idx += 1
            self.counter += 1

            if self.s_idx == self.g and not self.s_filled:
                self.s_idx = 0
                self.s_filled = True

            # TODO learning
        else:
            self.counter = 0

    def generate_input(self):
        """
        Generates a random input vector which consists of `k * n`
        values which are either 0 or 1.

        :return: Random input vector
        """

        return np.random.randint(0, 2, self.k * self.n)

    def get_key(self):
        """
        Key in permutation parity machine is the state vector.
        This function returns the key as a vector of bytes.

        :return: Key
        """

        key = 0

        for val in self.s:
            key = (key << 1) | int(val)

        return key.to_bytes(math.ceil(self.g / 8), byteorder='big')

    def synchronized(self):
        """
        Returns whether the two parity machines are synchronized.

        :return: True if the two parity machines are synchronized, False otherwise
        """

        return self.counter >= self.g
