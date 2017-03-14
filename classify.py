# Load pickled data
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import tensorflow as tf

class Data:
    def __init__(self):
        training_file = 'data/train.p'
        validation_file= 'data/valid.p'
        testing_file = 'data/test.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

    def render_data(self):
        image_with_label = zip(self.X_train, self.y_train)
        seen_labels = set()

        fig = plt.figure(figsize=(200, 200))
        total_unique_labels = len(set(self.y_train))
        unique_rows = total_unique_labels // 5 + 1

        grid = ImageGrid(fig, 151,  # similar to subplot(141)
                        nrows_ncols=(unique_rows, 5),
                        axes_pad=0.05,
                        label_mode="1",
                        )

        i = 0
        for i_l in image_with_label:
            img, label = i_l
            if label not in seen_labels:
                im = grid[i].imshow(img)
                seen_labels.add(label)
                i += 1

        plt.show()

def LeNet(x):
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1:
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))

    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation Layer
    conv1 = tf.nn.relu(conv1)

    # Max Pooling
    conv1 = tf.nn.max_pool(conv1, ksize[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2:
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation Layer
    conv2 = tf.nn.relu(conv2)

    # Max Pooling
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully Connected Layer
    fc0 = flatten(conv2)

    # Layer 3 - Fully Connected
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation
    fc1 = tf.nn.relu(fc1)

    # Layer 4 : Fully Connected
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def train():
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))

    one_hot_y = tf.one_hot(y, 10)

    rate = 0.001

    logits = Lenet(x)

    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation     = tf.reduce_mean(cross_entropy)
    optimizer          = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    return saver

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def main():
    data = Data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        saver = train(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(data.X_train, data.y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = data.X_train[offset:end], data.y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(data.X_validation, data.y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")


if __name__ == "__main__":
    main()
