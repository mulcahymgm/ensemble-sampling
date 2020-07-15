import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda, Layer
from keras.losses import binary_crossentropy, mean_squared_error, mse
from keras.models import Model, Sequential
from keras.optimizers import Adagrad, Adam


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    Taken from https://keras.io/examples/generative/vae/
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE:
    """
    A simple Variational AutoEncoder
    """

    def __init__(self, intermediate_dim=8, latent_dim=2):
        self._intermediate_dim = intermediate_dim
        self._latent_dim = latent_dim

    def build_encoder(self, input_shape, activation="relu"):

        inputs = Input(shape=input_shape, name="encoder_input")
        x = Dense(units=self._intermediate_dim, activation=activation)(inputs)

        z_mean = Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = Dense(self._latent_dim, name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

        encoder.summary()
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        return (encoder, inputs)

    def build_decoder(
        self, original_dim, intermediate_activation="relu", final_activation="tanh"
    ):

        latent_inputs = Input(shape=(self._latent_dim,), name="z_sampling")

        x = Dense(units=self._intermediate_dim, activation=intermediate_activation)(
            latent_inputs
        )

        outputs = Dense(units=original_dim, activation=final_activation)(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name="decoder")

        decoder.summary()
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        return (decoder, outputs)

    def rec_loss(self, y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)
        reconstruction_loss *= self._num_features
        return K.mean(reconstruction_loss)

    def kl_loss(self, y_true, y_pred):
        z_mean = self._vae.get_layer("encoder").get_layer("z_mean").output
        z_log_var = self._vae.get_layer("encoder").get_layer("z_log_var").output
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(kl_loss)

    def vae_loss(self, y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)
        reconstruction_loss *= self._num_features

        z_mean = self._vae.get_layer("encoder").get_layer("z_mean").output
        z_log_var = self._vae.get_layer("encoder").get_layer("z_log_var").output
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return K.mean(reconstruction_loss + kl_loss)

    def build_vae(self, encoder, decoder, inputs, outputs, optimizer="adam"):
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        self._vae = Model(inputs, outputs, name="vae_mlp")
        self._vae.summary()

        self._vae.compile(
            optimizer=optimizer,
            loss=self.vae_loss,
            metrics=[self.rec_loss, self.kl_loss],
        )

        # plot_model(self._vae, to_file='vae_mlp.png', show_shapes=True)

    def fit(self, X, y=None, epochs=30, batch_size=32):
        """
        Might use y for embedding
        """

        self._num_features = X.shape[1]
        self._encoder, inputs = self.build_encoder((self._num_features,))
        self._decoder, outputs = self.build_decoder(self._num_features)
        self.build_vae(self._encoder, self._decoder, inputs, outputs)

        self._history = self._vae.fit(X, X, epochs=epochs, batch_size=batch_size)

        return self

    def calculate_reconstruction_error(self, X):
        # predict the output for the X
        vae_predictions = self._vae.predict(X)

        # calculate the reconstruction error based on the mean square error of the difference
        # between the predictions and training values themselves
        vae_reconstruction_error = np.mean(np.power(X - vae_predictions, 2), axis=1)

        print("Average Reconstruction Error =", np.mean(vae_reconstruction_error))
        return vae_reconstruction_error

    def generate_samples(self, n_samples=64):
        """
        Generates new random but very realistic features using
        a trained generator model

        Params:
            class_for: Int - features for this class
            n_samples: Int - how many samples to generate
        """

        noise = np.random.normal(0, 1, (n_samples, self._latent_dim))
        return self._decoder.predict([noise])
