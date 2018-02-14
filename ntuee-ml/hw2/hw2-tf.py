import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from tensorflow.contrib.learn.python import SKCompat

import numpy as np
from numpy import genfromtxt
import random

# 取random data作為training data
def getBatchData(seed_index, batch_size):
    x_batch = []
    y_batch = []
    np.random.shuffle(seed_index)
    for i in range(batch_size):
        x_batch.append(X_train[seed_index[i]])
        y_batch.append(Y_train[seed_index[i]])
    return np.array(x_batch), np.array(y_batch)

seed_index=np.arange(1024)

X_train = genfromtxt('X_train', delimiter=',')
Y_train = genfromtxt('Y_train', delimiter=',')
X_test = genfromtxt('X_test', delimiter=',')

## tree method不需做normalize

X_train = np.array(X_train[1:]).astype('float32')
Y_train = np.array(Y_train[1:]).astype('float32')
X_test = np.array(X_test[1:]).astype('float32')

dim = len(X_train[0])

# Parameters
num_steps = 500
batch_size = 1024
num_classes = 2
num_features = 106
num_trees = 1000
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()
# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    s = random.randint(1,32561)
    # batch_x = X_train[s:s+2047:2]
    # batch_y = Y_train[s:s+2047:2]
    print(seed_index,batch_size)
    batch_x, batch_y = getBatchData(seed_index, batch_size)

    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

classifier =SKCompat(tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams))

classifier.fit(x=X_train, y=Y_train)

y_out = classifier.predict(x=X_test)

with open('Y_test1', 'w') as f:
    f.write('id,label\n')
    for i, l in enumerate(y_out['classes']):
        f.write('%d,%d\n' % ((i+1),l))