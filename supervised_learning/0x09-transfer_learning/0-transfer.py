#!/usr/bin/env python3
"""Contains an script that trains a convolutional neural network
to classify the CIFAR 10 dataset
"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """pre-processes the data for your model

    Args:
        X: Array containing the CIFAR 10 data
        Y: Array containing the CIFAR 10 labels for X
    """
    X_p = K.applications.resnet50.preprocess_input(x=X)
    Y_p = K.utils.to_categorical(y=Y)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    b_model = K.applications.InceptionResNetV2(weights="imagenet",
                                               input_shape=(288, 288, 3),
                                               include_top=False)
    b_model.summary()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)
    inputs = K.Input(shape=(32, 32, 3))
    images = K.layers.Lambda(lambda image: K.backend.resize_images(
        x=image,
        height_factor=288/32,
        width_factor=288/32,
        data_format='channels_last'
    )
    )(inputs)
    layer = b_model(images, training=False)
    pool = K.layers.GlobalAveragePooling2D()(layer)
    fc1 = K.layers.Dense(units=512)(pool)
    batch1 = K.layers.BatchNormalization()(fc1)
    activation1 = K.layers.Activation('relu')(batch1)
    dropout1 = K.layers.Dropout(0.3)(activation1)

    fc2 = K.layers.Dense(units=512)(dropout1)
    batch2 = K.layers.BatchNormalization()(fc2)
    activation2 = K.layers.Activation('relu')(batch2)
    dropout2 = K.layers.Dropout(0.3)(activation2)

    fc3 = K.layers.Dense(10, activation='softmax')(dropout2)

    model = K.Model(inputs=inputs, outputs=fc3)
    b_model.trainable = False
    for layer in b_model.layers:
        layer.trainable = False

    optimizer = K.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=Y_train,
              batch_size=300,
              shuffle=True,
              epochs=2,
              validation_data=(X_test, Y_test),
              verbose=True)
    for layer in b_model.layers[:498]:
        layer.trainable = False
    for layer in b_model.layers[498:]:
        layer.trainable = True
    optimizer = K.optimizers.Adam(1e-5)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    history = model.fit(x=X_train,
                        y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=1,
                        epochs=5,
                        verbose=True)
    model.save('cifar10.h5')
