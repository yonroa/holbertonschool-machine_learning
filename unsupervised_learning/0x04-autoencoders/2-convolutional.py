#!/usr/bin/env python3
"""This module contains the function 'autoencoder'"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input_encoder = keras.Input(shape=input_dims)
    encoded = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu', padding='same')(input_encoder)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    for i in range(1, len(filters)):
        encoded = keras.layers.Conv2D(
            filters[i], (3, 3), activation='relu', padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    input_decoder = keras.Input(shape=(latent_dims))

    prev_layer = input_decoder
    for i in range(len(filters) - 1, 0, -1):
        hidden_layer = keras.layers.Conv2D(filters[i],
                                           activation='relu',
                                           kernel_size=(3, 3),
                                           padding='same')
        prev_layer = hidden_layer(prev_layer)
        upsample_layer = keras.layers.UpSampling2D((2, 2))
        prev_layer = upsample_layer(prev_layer)

    last_layer = keras.layers.Conv2D(filters[0],
                                     kernel_size=(3, 3),
                                     padding='valid',
                                     activation='relu')
    prev_layer = last_layer(prev_layer)
    upsample_layer = keras.layers.UpSampling2D((2, 2))
    prev_layer = upsample_layer(prev_layer)

    output_layer = keras.layers.Conv2D(input_dims[2],
                                       activation='sigmoid',
                                       kernel_size=(3, 3),
                                       padding='same')
    decoder_outputs = output_layer(prev_layer)

    encoder = keras.Model(inputs=input_encoder, outputs=encoded)
    decoder = keras.Model(inputs=input_decoder, outputs=decoder_outputs)
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
