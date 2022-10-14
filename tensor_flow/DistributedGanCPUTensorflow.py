import os

import numpy as np
import ray
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as k
from tensorflow.keras.preprocessing import image

from helperUtils.tf_utils import plot_images

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Extnesion of https://github.com/bbondd/DistributedGAN
class Data(object):
    def load_mnist(self):
        (real_data, _), (_, _) = tf.keras.datasets.mnist.load_data()
        # Make it 3D dataset
        real_data = real_data[:5000].astype('float32')
        real_data = np.reshape(real_data, [-1, C.image_width, C.image_width, 1])
        # Standardize data : 0 to 1
        real_data = real_data.astype('float32') / 255

        self.real_data = real_data
        return real_data

    def __init__(self):
        self.real_data = None
        self.load_mnist()


class C(object):

    def make_fixed_discriminator(data_shape=[28, 28, 1, ]):
        # Build the network
        dis_input = tf.keras.layers.Input(data_shape)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(dis_input)
        x = tf.keras.layers.Conv2D(32, kernel_size=[5, 5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=[5, 5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(128, kernel_size=[5, 5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=[5, 5], strides=1, padding='same')(x)

        # Flatten the output and build an output layer
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # Build Model
        discriminator = tf.keras.models.Model(dis_input, x, name='discriminator')

        return discriminator


    def make_fixed_generator(image_size=28, input_size=100):
        # Build an input layer
        gen_input = tf.keras.layers.Input(shape=(input_size,))

        # Increase dimensions and resize to 3D to feed it to Conv2DTranspose layer
        x = tf.keras.layers.Dense(7 * 7 * 128)(gen_input)
        x = tf.keras.layers.Reshape((7, 7, 128))(x)

        # Use ConvTranspose
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=[5, 5], strides=2, padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=[5, 5], strides=2, padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(32, kernel_size=[5, 5], strides=1, padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2DTranspose(1, kernel_size=[5, 5], strides=1, padding='same')(x)

        # Output layer for Generator
        x = tf.keras.layers.Activation('sigmoid')(x)

        # Build model using Model API
        generator = tf.keras.models.Model(gen_input, x, name='generator')

        return generator

    image_width = 28
    noise_dimension = 100

    gan_size = 3
    batch_size = 64
    learning_rate = 2e-4
    decay = 6e-8
    path = '../results'
    generate_image_size = 16


@ray.remote(num_cpus = 1)
class Gan(object):
    def __init__(self, real_data, generator, base_discriminator, learning_rate):
        self.real_data = real_data
        optimizer = k.optimizers.Adam(learning_rate)

        discriminator = tf.keras.models.Model(inputs=base_discriminator.inputs,
                                              outputs=base_discriminator.outputs)

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, decay=C.decay)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

        frozen_discriminator = tf.keras.models.Model(inputs=base_discriminator.inputs,
                                                     outputs=base_discriminator.outputs)
        # Freeze the weights of discriminator during adversarial training
        frozen_discriminator.trainable = False

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate * 0.5, decay=C.decay * 0.5)
        # Adversarial = generator + discriminator
        adversarial = tf.keras.models.Model(generator.input,
                                            frozen_discriminator(generator.output))

        adversarial.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])

        self.discriminator = discriminator
        self.generator = generator
        self.adversarial = adversarial


    def train_on_batch_index(self, batch_index):
        noise = np.random.uniform(-1, 1, [len(batch_index), C.noise_dimension]).astype('float32')
        fake_data = self.generator.predict(noise)
        real_data = self.real_data[batch_index]

        X = np.concatenate((real_data, fake_data))

        y_real = np.ones([len(batch_index), 1])
        y_fake = np.zeros([len(batch_index), 1])
        y = np.concatenate((y_real, y_fake))
        self.discriminator.train_on_batch(X, y)

        X = noise = np.random.uniform(-1, 1, [len(batch_index), C.noise_dimension]).astype('float32')
        y = np.ones([len(batch_index), 1])

        self.adversarial.train_on_batch(X, y)

    def get_generator_weights(self):
        return self.generator.get_weights()

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)

    def get_discriminator_weights(self):
        return self.discriminator.get_weights()

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)


    def get_sample_images(self):
        noise = np.random.uniform(-1, 1, [C.generate_image_size, C.noise_dimension])
        image_arrays = self.generator.predict(x=noise)

        return image_arrays

@ray.remote(num_cpus = 1)
class DistributedGan(object):

    def __init__(self, distribution_size, real_data, generator, discriminator):
        self.discriminator = discriminator
        self.generator = generator
        self.gans = [Gan.remote(real_data[np.random.choice(len(real_data), len(real_data))],
                                k.models.clone_model(generator),
                                k.models.clone_model(discriminator),
                                C.learning_rate * distribution_size) for _ in range(distribution_size)]

    def get_generator_weights(self):
        weights_set = ray.get([gan.get_generator_weights.remote() for gan in self.gans])
        return np.mean(weights_set, axis=0)

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)
        ray.get([gan.set_generator_weights.remote(weights) for gan in self.gans])

    def get_discriminator_weights(self):
        weights_set = ray.get([gan.get_discriminator_weights.remote() for gan in self.gans])
        return np.mean(weights_set, axis=0)

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)
        ray.get([gan.set_discriminator_weights.remote(weights) for gan in self.gans])

    def train_on_batch_index(self, batch_index):
        ray.get([gan.train_on_batch_index.remote(batch_index) for gan in self.gans])
        self.set_generator_weights(self.get_generator_weights())
        self.set_discriminator_weights(self.get_discriminator_weights())

    def save_generator(self, directory_path, iteration_number):
        generator_path = directory_path + '/generator'
        os.makedirs(generator_path, exist_ok=True)

        self.generator.save_weights(generator_path + '/weights.h5')
        self.generator.save(generator_path + '/model')
        with open(generator_path + '/architecture.json', 'w') as f:
            f.write(self.generator.to_json())
        k.utils.plot_model(self.generator, generator_path + '/graph.png', show_shapes=True)

        image_path = directory_path + '/image'

        image_arrays = ray.get(self.gans[0].get_sample_images.remote())

        os.makedirs(image_path, exist_ok=True)

        for j in range(len(image_arrays)):
            image.save_img(x=image_arrays[j],
                           path=image_path + '/iteration%d num%d.png' % (iteration_number, j))

        plot_images(image_arrays, generator_path, iteration_number)

        plt.show()

    def save_discriminator(self, directory_path):
        discriminator_path = directory_path + '/discriminator'
        try:
            os.makedirs(discriminator_path)
        except FileExistsError:
            pass

        self.discriminator.save_weights(discriminator_path + '/weights.h5')
        self.discriminator.save(discriminator_path + '/model')

        with open(discriminator_path + '/architecture.json', 'w') as f:
            f.write(self.discriminator.to_json())
        k.utils.plot_model(self.discriminator, discriminator_path + '/graph.png', show_shapes=True)



class MultiDistributedGan(object):
    def __init__(self, real_data, distribution_size, generators, discriminators):
        self.same_generator_gans_group = [[] for _ in range(len(generators))]
        self.same_discriminator_gans_group = [[] for _ in range(len(discriminators))]
        self.gans = []

        for i in range(len(generators)):
            for j in range(len(discriminators)):
                distributed_gan = DistributedGan.remote(distribution_size,
                                                 real_data[np.random.choice(len(real_data), len(real_data))],
                                                 k.models.clone_model(generators[i]),
                                                 k.models.clone_model(discriminators[j]))

                self.gans.append(distributed_gan)
                self.same_generator_gans_group[i].append(distributed_gan)
                self.same_discriminator_gans_group[j].append(distributed_gan)

    def get_generators_weights(self):
        generators_weights = []
        for same_generator_gans in self.same_generator_gans_group:
            weights_set = [ray.get(gan.get_generator_weights.remote()) for gan in same_generator_gans]
            generators_weights.append(np.mean(weights_set, axis=0))

        return generators_weights

    def set_generators_weights(self, generators_weights):
        for same_generator_gans, generator_weights in zip(self.same_generator_gans_group, generators_weights):
            [ray.get(gan.set_generator_weights.remote(generator_weights)) for gan in same_generator_gans]

    def get_discriminators_weights(self):
        discriminators_weights = []
        for same_discriminator_gans in self.same_discriminator_gans_group:
            weights_set = []
            for same_discriminator_gan in same_discriminator_gans:
                weights_set.append(ray.get(same_discriminator_gan.get_discriminator_weights.remote()))
            discriminators_weights.append(np.mean(np.array(weights_set), axis=0))

        return discriminators_weights

    def set_discriminators_weights(self, discriminators_weights):
        for same_discriminator_gans, discriminator_weights \
                in zip(self.same_discriminator_gans_group, discriminators_weights):
            [gan.set_discriminator_weights.remote(discriminator_weights) for gan in same_discriminator_gans]

    def train_on_batch_index(self, batch_index):
        [gan.train_on_batch_index.remote(batch_index) for gan in self.gans]
        self.set_generators_weights(self.get_generators_weights())
        self.set_discriminators_weights(self.get_discriminators_weights())


    def save(self, iteration_number):
        for i in range(len(self.same_generator_gans_group)):
            distributed_gan = self.same_generator_gans_group[i][0]
            ray.get(distributed_gan.save_generator.remote(C.path + '/generator %d' % i, iteration_number))

        for i in range(len(self.same_discriminator_gans_group)):
            distributed_gan = self.same_discriminator_gans_group[i][0]
            ray.get(distributed_gan.save_discriminator.remote(C.path + '/discriminator %d' % i))


def main():
    print('redis server :')
    ray.init()

    real_data = Data().real_data

    make_generator_function = C.make_fixed_generator
    make_discriminator_function = C.make_fixed_discriminator

    generator_size = 1

    discriminator_size = 1

    distribution_size = 2  # Change this for each client

    iteration_size = 100

    generators = [make_generator_function(image_size=28, input_size=C.noise_dimension) for _ in range(generator_size)]
    discriminators = [make_discriminator_function(data_shape=(28, 28, 1,)) for _ in range(discriminator_size)]

    multi_distributed_gan = MultiDistributedGan(real_data, distribution_size, generators, discriminators)


    for i in range(iteration_size):
        if i % 10 == 0:
            multi_distributed_gan.save(i)
            print('iteration', i)

        batch_indexes = np.array_split(np.random.permutation(len(real_data)), int(len(real_data) / C.batch_size))
        for batch_index in batch_indexes:

            multi_distributed_gan.train_on_batch_index(batch_index)


if __name__ == '__main__':
    main()
