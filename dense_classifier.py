from __future__ import absolute_import, division

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, Activation, PReLU
from tensorflow.keras import regularizers

# method to return the dense classifier
def get_classifier(input_length, multi_label_classification, layer_sizes=[100], output_size=2):
    model = Sequential()
    model.add(Flatten(input_shape=input_length))
    model.add(BatchNormalization())
    for layer_size in layer_sizes:
        if layer_size < 1:
            model.add(Dropout(layer_size))
        else:
            model.add(Dense(layer_size))
            model.add(BatchNormalization())
            model.add(Activation(PReLU()))

    if multi_label_classification:
        model.add(Dense(output_size, activation='sigmoid', name="predictions",
                  kernel_regularizer=regularizers.l2(0.05)))
    else:
        model.add(Dense(output_size, activation='softmax', name="predictions",
                  kernel_regularizer=regularizers.l2(0.05)))

    return model
