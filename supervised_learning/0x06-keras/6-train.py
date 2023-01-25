#!/usr/bin/env python3
"""Contain the function 'train_model'"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent, analyze validaiton,
    and train the model using early stoppingdata

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
    """
    calls = []
    if validation_data and early_stopping:
        early = K.callbacks.EarlyStopping(patience=patience)
        calls.append(early)

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=calls)
