import tensorflow as tf
from tensorflow.contrib import layers



class NN:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(210,160,3))
        self.targetQ = tf.placeholder(tf.float32, shape=(4))
        features = [tf.reshape(self.input, [-1])]
        features = features[0:210 * 160]


        regularizer = layers.l2_regularizer(0.01)

        # Structure
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 550, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.dropout(features)
        #features = layers.fully_connected(features, 1520, weights_regularizer=regularizer)
        #features = layers.fully_connected(features, 2010)
        #features = layers.bias_add(features, regularizer=regularizer)
        #features = layers.fully_connected(features, 700, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 200, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 100, weights_regularizer=regularizer)
        features = layers.bias_add(features, regularizer=regularizer)
        features = layers.fully_connected(features, 4, weights_regularizer=regularizer)

        self.predict = features[0]

        self.loss = tf.reduce_sum(tf.square(self.targetQ - self.predict))

        trainer = tf.train.AdamOptimizer()
        self.train = trainer.minimize(self.loss)
