from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_autoencoder(n_dimensions, size_encoder, weights=False):

    input_layer = Input(shape=(n_dimensions,))
    encoded = Dense(size_encoder, activation='relu')(input_layer)
    decoded = Dense(n_dimensions, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    encoder = Model(inputs=input_layer, outputs=encoded)

    encoded_input = Input(shape=(size_encoder,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    if weights:
        autoencoder.set_weights(weights)

    autoencoder.compile(loss='mse', optimizer='adam')

    return autoencoder, encoder, decoder
