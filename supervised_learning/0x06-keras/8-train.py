#!/usr/bin/env python3
"""Contain the function 'train_model'"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent, analyze validaiton,
    and train the model using early stoppingdata, also train the model with
    learning rate decay

    Args:
        network: model to train
        data: Array that contains the input data
        labels: One-hot array containing the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean that determines if output should be printed
            during training
        shuffle: boolean that determines whether to shuffle the batches
            every epoch
        validation_data: data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping
            should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean that indicates whether learning rate
            decay should be used
        alpha: initial learning rate
        decay_rate: decay rate
        save_best: boolean indicating whether to save the model after each
            epoch if it is the best
        filepath: file path where the model should be saved
    """
    calls = []
    if validation_data and early_stopping:
        early = K.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          patience=patience)
        calls.append(early)
    if validation_data and learning_rate_decay:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)
        learn = K.callbacks.LearningRateScheduler(schedule=schedule,
                                                  verbose=1)
        calls.append(learn)

    checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                             save_best_only=save_best)
    calls.append(checkpoint)

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=calls)