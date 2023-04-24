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
        self.buffer = np.empty(self.g, dtype=np.int8)

        self.buffer_idx = 0
        self.buffer_filled = False
        self.counter = 0

    def __call__(self, x, pi):
        """
        The output of the hidden layer is the sum of XOR products of
        the input and the weights. Weight vector is created by
        permuting the state vector. The output of the hidden layer is
        then fed into a threshold function which outputs 1 if the sum
        is greater than `n / 2` and 0 otherwise. The output of the
        network is the product of the outputs of the hidden layers.

        :param x: Input vector
        :param pi: Permutation vector
        :return: Output of the parity machine
        """

        W = self.s[pi].reshape(self.k, self.n)
        X = x.reshape(self.k, self.n)

        h = np.bitwise_xor(X, W).sum(axis=1)
        sigma = np.where(h > self.n / 2, 1, 0)
        tau = np.bitwise_xor.reduce(sigma)

        self.sigma0 = sigma[0]
        self.tau = tau

        return tau

    def update(self, tau):
        """
        Performs the update of the parity machine. The update is
        performed only if the output of the networks are equal.
        The update consists of setting the current output of the
        first hidden neuron to the buffer and incrementing the
        buffer index. If the index is greater than or equal to the
        buffer size, the state vector is filled with the contents
        of the buffer and the buffer index is reset to 0.

        :param tau:
        :return:
        """

        if tau == self.tau:
            self.buffer[self.buffer_idx] = self.sigma0

            self.buffer_idx += 1
            self.counter += 1

            if self.buffer_idx == self.g:
                self.buffer_filled = True
                self.s = np.copy(self.buffer)
                self.buffer_idx = 0
        else:
            self.counter = 0

    def generate_input(self):
        """
        Generates a random input vector and a random permutation
        vector  which consists of `k * n` values which are either
        0 or 1.

        :return: Random input vector and permutation vector
        """

        x = np.random.randint(0, 2, self.k * self.n)
        pi = np.random.permutation(self.g)[:self.k * self.n]

        return x, pi

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
        Synchronization is achieved when the buffer is filled and
        the counter is greater than or equal to the buffer size which
        means that the last `g` outputs of both networks were equal.

        :return: True if the two parity machines are synchronized, False otherwise
        """

        return self.buffer_filled and self.counter >= self.g
