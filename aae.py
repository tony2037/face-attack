from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers import Add, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.preprocessing.image import save_img
import keras.backend as K

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

import numpy as np

from loader import Loader

class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 200
        self.img_cols = 180
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        image_a = Input(shape=self.img_shape)
        image_b = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr_a = self.encoder(image_a)
        encoded_repr_b = self.encoder(image_b)
        reconstructed_img = self.decoder(encoded_repr_a)

        # For the adversarial_autoencoder model we will only train the generator
        # self.discriminator.trainable = False

        # The discriminator determines the validity of the reconstruct image
        validity = self.discriminator(reconstructed_img)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model([image_a, image_b], [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.01, 0.99],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)

        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        def sample(x):
            return K.random_normal(K.shape(x[0])) * K.exp(x[1] / 2)
        sample_repr = Lambda(sample)([mu, log_var])
        latent_repr = Add()([mu, sample_repr])

        return Model(img, latent_repr)

    def build_decoder(self):

        z_a = Input(shape=(self.latent_dim,))
        #z_b = Input(shape=(self.latent_dim,))
        #img_a = Input(shape=self.img_shape)

        #z = Concatenate(-1)([z_a, z_b])
        out = Dense(512)(z_a)
        out = LeakyReLU(alpha=0.2)(out)
        out = Dense(512)(out)
        out = LeakyReLU(alpha=0.2)(out)
        out = Dense(np.prod(self.img_shape), activation='relu')(out)
        mask = Reshape(self.img_shape)(out)
        #def overlap(x):
        #    return (x[0] + x[1]) / 2
        #img = Lambda(overlap)([img_a, mask])
        
        return Model(z_a, mask)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        def add_epsilon(x):
            return x + K.epsilon()
        model.add(Lambda(add_epsilon))

        img = Input(shape=self.img_shape)
        validity = model(img)

        model.summary()
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        loader = Loader('faces94/malestaff/voudcx', 'faces94/malestaff/tony')
        img_a, img_b = loader.get_all()
        assert img_a.shape[0] == batch_size

        # Adversarial ground truths
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = np.zeros((batch_size, 1), dtype=np.float32)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Train the discriminator
            
            img_fake, _ = self.adversarial_autoencoder.predict([img_a, img_b])
            save_img('fake.jpg', img_fake[10])
            history_real = self.discriminator.fit(img_b, valid, batch_size=batch_size)
            history_fake = self.discriminator.fit(img_a, fake, batch_size=batch_size)
            d_loss_real = history_real.history['loss']
            d_loss_fake = history_fake.history['loss']
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            history_g = self.adversarial_autoencoder.fit([img_a, img_b], [img_b, valid], batch_size=batch_size)
            g_loss = history_g.history['loss']
            print(history_g.history.keys())

            # Plot the progress
            #print ("%d [D loss: %f] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch, img_a, img_b)

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20000, batch_size=20, sample_interval=200)
