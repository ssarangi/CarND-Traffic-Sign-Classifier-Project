import tensorflow as tf
import numpy as np

import json

class Node:
    def __init__(self, type, filter_size, stride, padding, activation_func, maxpool, connects_to):
        self.type = type
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation_func = activation_func
        self.maxpool = maxpool
        self.connects_to = connects_to

class Architecture:
    def __init__(self, filename):
        json = json.load(filename)
        nodes = []

    def _topological_sort(self):
        pass

def main():
    arch = Architecture("lenet.json")

if __name__ == "__main__":
    main()
