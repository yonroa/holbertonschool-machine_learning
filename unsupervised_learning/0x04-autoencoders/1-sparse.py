#!/usr/bin/env python3
"""This module contains the function 'autoencoder'"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder"""
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(encoded)
    encoded = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(encoded)

    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(inputs=input_encoder, outputs=encoded)
    decoder = keras.Model(inputs=input_decoder, outputs=decoded)
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
