#!/usr/bin/env python3
"""Contains the function 'train'"""

import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                tloss, taccuracy = sess.run([loss, accuracy],
                                            feed_dict={x: X_train, y: Y_train})
                print("\tTraining Cost: {}".format(tloss))
                print("\tTraining Accuracy: {}".format(taccuracy))
                vloss, vaccuracy = sess.run([loss, accuracy],
                                            feed_dict={x: X_valid, y: Y_valid})
                print("\tValidation Cost: {}".format(vloss))
                print("\tValidation Accuracy: {}".format(vaccuracy))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        saver = tf.train.Saver()
        return saver.save(sess, save_path)