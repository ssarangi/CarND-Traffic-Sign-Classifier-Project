import tensorflow as tf
import numpy as np

import json

class Maxpool:
    def __init__(self, size, strides):
        self.size = size
        self.strides = strides

    def get_tensor(self, inputs):
        return tf.layers.max_pooling2d(inputs=inputs, self.size, self.strides)

    def __str__(self):
        return "Maxpool: (%s) -> (%s)" % (size, strides)

    __repr__ = __str__

class CNN:
    def __init__(self, filters, kernel_size, strides, padding, activation_func):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_func = activation_func
        self.maxpool = None
        self.connects_to = None

    def _activation_func(self):
        if activation_func == "relu":
            return tf.nn.relu

        return None

    def get_tensor(self, inputs):
        tensor = tf.layers.conv2d(
            inputs=inputs,
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=self._activation_func()
        )

        if self.maxpool is not None:
            tensor = self.maxpool.get_tensor(tensor)

        return tensor

    def __str__(self):
        return self.type

    __repr__ = __str__

class Architecture:
    def __init__(self, filename):
        fp = open(filename, 'r')
        s = fp.read()
        arch_str = json.loads(s)

        node_map = {}

        self.nodes = []

        # Create the nodes first.
        for key in arch_str:
            k = arch_str[key]
            if k['type'] == 'cnn':
                node = CNN(k['filters'], k['filter_size'], k['stride'], k['padding'], k['activation_func'], k['maxpool'])
                # Setup the Maxpool Node
                node.maxpool = Maxpool(k['maxpool']['size'], k['maxpool']['strides'])
                node_map[key] = node
                self.nodes.append(node)

        # Create the connections from the node
        for key in arch_str:
            k = arch_str[key]
            src_node = node_map[key]
            if k['connects_to'] is not None:
                target_node = node_map[k['connects_to']]
                src_node.connects_to = target_node

        self._topological_sort()

    def _get_entry_node(self):
        return self.nodes[0]

    def _topological_sort(self):
        # Do a topological sort to find out the first node we need
        pass

    def generate_graph(features, labels):
        entry_node = self._get_entry_point()

        # Now at this point we can recursively go through and find the tensor
        


def main():
    arch = Architecture("lenet.json")

if __name__ == "__main__":
    main()