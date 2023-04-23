from abc import ABC, abstractmethod

import numpy as np


class ParityMachine(ABC):
    def __init__(self, k, n, seed=None):
        """
        Base class for parity machines. A parity machine is a neural network
        with a single hidden layer of `k` neurons, each of which has `n` inputs.
        The output of the hidden layer is then fed into a single neuron
        which computes the parity of the hidden layer's output.

        :param k: Number of neurons in the hidden layer
        :param n: Number of inputs to each neuron in the hidden layer
        :param seed: Seed for the random number generator
        """

        assert k > 0, "Number of neurons must be greater than 0"
        assert n > 0, "Number of inputs must be greater than 0"

        self.k = k
        self.n = n

        np.random.seed(seed)

    @abstractmethod
    def __call__(self, x):
        """
        Computes the output of the parity machine for a given input.

        :param x: Input vector
        :return: Output of the parity machine
        """

        pass

    @abstractmethod
    def update(self, tau):
        """
        Updates the weights of the parity machine.

        :param tau: Output of the other parity machine
        """

        pass

    @abstractmethod
    def generate_input(self):
        """
        Generates a random input vector.

        :return: Random input vector
        """

        pass

    @abstractmethod
    def get_key(self):
        """
        Returns the key based on the weights or state of the parity machine.

        :return: Key
        """

        pass

    @abstractmethod
    def synchronized(self):
        """
        Checks if the parity machines are synchronized.

        :return: `True` if synchronized, `False` otherwise
        """

        pass
