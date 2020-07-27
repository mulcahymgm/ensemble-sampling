import numpy as np

# Embedding layers
# Merge layers
# Normalization Layers
# Core layers
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    multiply,
)

# Keras models
from keras.models import Model, Sequential

# keras optimizers
from keras.optimizers import SGD, Adam, RMSprop


class CGAN:
    """An implementation of a conditional Generative Adversarial Network (Mirza & Osindero, 2014)

    This implementation is based on the following web article, intm
    https://medium.com/@jscriptcoder/data-augmentation-using-conditional-gan-cgan-d5e8d33ad032,
    which is itself based on the following implementation,
    https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py

    References:
    Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. Retrieved from https://arxiv.org/abs/1411.1784

    Parameters
    ----------
    latent_dim : int, default = 100
        The size of a noise sample generated in training.

    """

    def __init__(self, latent_dim=50):
        self.latent_dim = latent_dim

    def build_discriminator(
        self,
        optimizer=Adam(0.0002, 0.5),
        alpha=0.2,
        layer_size=16,
        number_layers=1,
        dropout=0.2,
    ):
        """Defines and compiles the discriminator model.

        This architecture has been inspired by: https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
        and adapted for this problem.

        Params:
            optimizer : optimizer, default Adam(0.0002, 0.5) - recommended values according to article
            alpha : float, default = 0.2
            layer_size : int, default = 16
            number_layers : int, default = 1
            dropout : float, default = 0.2
        """

        features = Input(shape=(self._num_features,))
        label = Input(shape=(1,), dtype="int32")

        # Using an Embedding layer is recommended by the papers
        # (mm - unfortunately doesn't say which papers)
        label_embedding = Flatten()(
            Embedding(self._num_classes, self._num_features)(label)
        )

        # We condition the discrimination of generated features
        inputs = multiply([features, label_embedding])

        x = inputs
        for _ in range(number_layers):
            x = Dense(layer_size)(x)
            x = LeakyReLU(alpha=alpha)(x)
        #    x = Dense(16)(x)
        #    x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(dropout)(x)

        valid = Dense(1, activation="sigmoid")(x)

        model = Model([features, label], valid)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        self.discriminator = model

    def build_generator(self, alpha=0.2, layer_size=16, number_layers=1, momentum=0.8):
        """Defines and compiles the generator model.

        This architecture has been inspired by: https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
        and adapted for this problem.

        Params:
            alpha : float, default = 0.2
            layer_size : int, default = 16
            number_layers : int, default = 1
            momentum : float, default = 0.8
        """

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")

        # Using an Embedding layer is recommended by the papers (mm - unfortunately doesn'tr say which papers)
        label_embedding = Flatten()(
            Embedding(self._num_classes, self.latent_dim)(label)
        )

        # We condition the generation of generated features
        inputs = multiply([noise, label_embedding])

        x = inputs
        for _ in range(number_layers):
            x = Dense(layer_size)(x)
            x = LeakyReLU(alpha=alpha)(x)
            x = BatchNormalization(momentum=momentum)(x)
        #    x = Dense(16)(x)
        #    x = LeakyReLU(alpha=0.2)(x)
        #    x = BatchNormalization(momentum=0.8)(x)

        features = Dense(self._num_features, activation="tanh")(x)

        # mm - note that the model is not compiled -
        # "Notice the generator model isn’t compiled.
        # This is because we’re not gonna directly train it.
        # It’s part of a combined model, and its weights will
        # change as we train the whole GAN mode"
        self.generator = Model([noise, label], features)

    def build_gan(self, optimizer=Adam(0.0002, 0.5)):
        """Defines and compiles GAN model.

        It basically chains Generator and Discriminator in an assembly line sort of way,
        where the input is the Generator's input. The Generator's output is the input of the
        Discriminator, which outputs the output of the whole GAN

        Params:
            optimizer=Adam(0.0002, 0.5) - recommended values
        """

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))

        features = self.generator([noise, label])

        valid = self.discriminator([features, label])

        # we freeze the discriminator's layers since we are only interested
        # in the generator and its learning
        # mm according to book this will only apply to Gan model
        self.discriminator.trainable = False

        model = Model([noise, label], valid)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        ### TODO: plot model
        ### plot_model(gan, to_file='cgan_model.png', show_shapes=True)
        ### maybe do it for generator and discriminator also

        self.gan = model

    def get_random_batch(self, X, y, batch_size):
        """
        Will return random batches of size batch_size

        Params:
            X: numpy array - features
            y: numpy array - classes
            batch_size: Int
        """

        idx = np.random.randint(0, len(X))

        X_batch = X[idx : idx + batch_size]
        y_batch = y[idx : idx + batch_size]

        return X_batch, y_batch

    def train_gan(
        self,
        gan,
        generator,
        discriminator,
        X,
        y,
        n_epochs=2000,
        batch_size=32,
        k=2,
        hist_every=10,
        log_every=100,
    ):
        """Trains discriminator and generator separately in batches of size batch_size.

        The training goes as follows:
        1. Discriminator is trainined with real features from training data
        2. Discriminator is training with fake features generated by the Generator
        3. GAN is trained, which will only change the Generators weights

        Params:
            gan: GAN Model
            generator: Generator Model
            discriminator: Discriminator Model
            X: numpy array - features
            y: numpy array - classes
            n_epochs: int
            batch_size: int
            k : int - hyperparameter representing number of times discriminator is trained in each training step (see GoodFellow et al)
            hist_every: int - will save the training loss and accuracy every hist_every epochs
            log_every: int - will output the loss and acurracy every log_every epochs

        Returns:
            loss_real_hist: List of Floats
            acc_real_hist: List of Floats
            loss_fake_hist: List of Floats
            acc_fake_hist: List of Floats
            loss_gan_hist: List of Floats
            acc_gan_hist: List of Floats

        """

        # half_batch = int(batch_size / 2)
        self.acc_real_hist_ = []
        self.acc_fake_hist_ = []
        self.acc_gan_hist_ = []
        self.loss_real_hist_ = []
        self.loss_fake_hist_ = []
        self.loss_gan_hist_ = []

        for epoch in range(n_epochs):

            # train discriminator with real values
            # mm - because these are the real values we set y_real to ones
            for _ in range(k):
                X_batch, labels = self.get_random_batch(X, y, batch_size)
                y_real = np.ones((X_batch.shape[0], 1))
                loss_real, acc_real = discriminator.train_on_batch(
                    [X_batch, labels], y_real
                )

                # train discriminator with fake values
                # mm - because these are the fake values we set y_fake to zeros
                noise = np.random.uniform(0, 1, (labels.shape[0], self.latent_dim))
                X_fake = generator.predict([noise, labels])
                y_fake = np.zeros((X_fake.shape[0], 1))
                loss_fake, acc_fake = discriminator.train_on_batch(
                    [X_fake, labels], y_fake
                )

            # train the generator via the GAN
            X_batch, labels = self.get_random_batch(X, y, batch_size)
            noise = np.random.uniform(0, 1, (labels.shape[0], self.latent_dim))
            y_gan = np.ones((labels.shape[0], 1))
            loss_gan, acc_gan = gan.train_on_batch([noise, labels], y_gan)

            if (epoch + 1) % hist_every == 0:
                self.acc_real_hist_.append(acc_real)
                self.acc_fake_hist_.append(acc_fake)
                self.acc_gan_hist_.append(acc_gan)
                self.loss_real_hist_.append(loss_real)
                self.loss_fake_hist_.append(loss_fake)
                self.loss_gan_hist_.append(loss_gan)

            if (epoch + 1) % log_every == 0:
                lr = "loss real: {:.3f}".format(loss_real)
                ar = "acc real: {:.3f}".format(acc_real)
                lf = "loss fake: {:.3f}".format(loss_fake)
                af = "acc fake: {:.3f}".format(acc_fake)
                lg = "loss gan: {:.3f}".format(loss_gan)
                ag = "acc gan: {:.3f}".format(acc_gan)

                print("{}, {} | {}, {} | {}, {}".format(lr, ar, lf, af, lg, ag))

        return (
            self.loss_real_hist_,
            self.acc_real_hist_,
            self.loss_fake_hist_,
            self.acc_fake_hist_,
            self.loss_gan_hist_,
            self.acc_gan_hist_,
        )

    def fit(
        self, X, y, n_epochs=2000, batch_size=32, k=2, hist_every=10, log_every=100,
    ):
        """Fits to the data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        n_epochs: int
        batch_size: int
        k : int - hyperparameter representing number of times discriminator is trained in each training step (see GoodFellow et al)
        hist_every: int - will save the training loss and accuracy every hist_every epochs
        log_every: int - will output the loss and acurracy every log_every epochs

        Returns
        -------
        self : Returns a trained model
        """

        self._num_features = X.shape[1]
        self._num_classes = len(set(y))
        self.build_discriminator()
        self.build_generator()
        self.build_gan()

        self.train_gan(
            self.gan,
            self.generator,
            self.discriminator,
            X,
            y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            k=k,
            hist_every=hist_every,
            log_every=log_every,
        )

        return self

    def score(self, X, y, sample_weight=None):
        """
           TODO - look at ways that a score could be created, e.g. maybe using Gaussian Parzen window in same
           manner as (Goodfellow et Al, 2014)

           Reference:
           Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … Bengio, Y. (2014). Generative Adversarial Networks. Retrieved from http://arxiv.org/abs/1406.2661


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : None
        """

        return None

    def generate_samples(self, class_for=1, n_samples=64):
        """Generates new random but very realistic features using a trained generator model

        Params:
            class_for: Int - features for this class
            n_samples: Int - how many samples to generate
        """

        noise = np.random.uniform(0, 1, (n_samples, self.latent_dim))
        label = np.full((n_samples,), fill_value=class_for)
        return self.generator.predict([noise, label])
