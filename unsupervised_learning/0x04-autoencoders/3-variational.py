#!/usr/bin/env python3
"""This module contains the function 'autoencoder'"""

import tensorflow.keras as keras


def sampling(args):
    """Sampling function"""
    mean, log_var = args
    epsilon = keras.backend.random_normal(
        shape=keras.backend.shape(mean), mean=0., stddev=1.)
    return mean + keras.backend.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(encoded)

    mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    z = keras.layers.Lambda(sampling)([mean, log_var])

    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(inputs=input_encoder, outputs=[z, mean, log_var])

    decoder = keras.Model(inputs=input_decoder, outputs=decoded)

    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)[0]))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
