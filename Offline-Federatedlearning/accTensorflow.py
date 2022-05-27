import os


from tensorflow import keras as k
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from helperUtils.tf_utils import *

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


class Data(object):
    def load_mnist(self):
        (real_data, real_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        # Make it 3D dataset
        real_data = real_data[:100].astype('float32')
        real_data = np.reshape(real_data, [-1, C.image_width, C.image_width, 1])
        # Standardize data : 0 to 1
        real_data = real_data.astype('float32') / 255

        self.real_data = real_data
        self.real_labels = tf.keras.utils.to_categorical(real_labels)
        return real_data, real_labels

    def __init__(self):
        self.real_data = None
        self.real_labels = None
        self.load_mnist()


class C(object):

    @staticmethod
    def make_fixed_discriminator(image_inputs, image_size=28):
        #Network parameters
        filter_size = 5
        num_filters = [32, 64, 128, 256]
        stride_size = [2, 2, 2, 1]

        #Build the network
        x = image_inputs
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=[5,5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=[5,5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(128, kernel_size=[5,5], strides=2, padding='same')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=[5,5], strides=1, padding='same')(x)

        #Flatten the output and build an output layer
        x = tf.keras.layers.Flatten()(x)

        #The main difference is that the model has two output layers :

        #The first is a single node with the sigmoid activation for predicting the real-ness of the image.
        out1 = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        #The second is multiple nodes, one for each class,
        #using the softmax activation function to predict the class label of the given image.
        out2 = tf.keras.layers.Dense(10, activation='softmax')(x)


        #Build Model
        discriminator = tf.keras.models.Model(image_inputs,
                                              [out1,out2], name='discriminator')

        return discriminator


    @staticmethod
    def make_fixed_generator(noise_inputs, label_inputs, image_size=28):

        # Concatenate both noise and labels
        x = tf.keras.layers.concatenate([noise_inputs, label_inputs], axis=1)

        # Increase dimensions and resize to 3D to feed it to Conv2DTranspose layer
        x = tf.keras.layers.Dense(7 * 7 * 128)(x)
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

        generator = tf.keras.models.Model([noise_inputs, label_inputs], x, name='generator')

        return generator

    image_width = 28
    noise_dimension = 100

    gan_size = 3
    batch_size = 64
    learning_rate = 2e-4
    decay = 6e-8
    path = './results'
    generate_image_size = 16
    num_labels = 10


    noise_inputs = tf.keras.layers.Input(shape=(noise_dimension,))
    label_inputs = tf.keras.layers.Input(shape=(10,))
    image_inputs = tf.keras.layers.Input(shape=(28, 28, 1,))

class Gan(object):
    def __init__(self, real_data,real_labels, generator, base_discriminator, learning_rate):
        self.real_data = real_data
        self.real_labels = real_labels
        discriminator = tf.keras.models.Model(inputs=base_discriminator.inputs,
                                              outputs=base_discriminator.outputs)

        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, decay=C.decay)
        discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
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

        adversarial.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                            optimizer=optimizer,
                            metrics=['accuracy'])

        self.discriminator = discriminator
        self.generator = generator
        self.adversarial = adversarial

    def train_on_batch_index(self, batch_index):
        noise = np.random.uniform(-1, 1, [len(batch_index), C.noise_dimension]).astype('float32')

        # 1. Get fake images from Generator
        fake_labels = np.eye(C.num_labels)[np.random.choice(C.num_labels, len(batch_index))]
        fake_data = self.generator.predict([noise, fake_labels])

        # 2. Get real images and labels from training set
        real_data = self.real_data[batch_index]
        real_labels = self.real_labels[batch_index]

        # 3. Prepare input for training Discriminator
        X = np.concatenate((real_data, fake_data))

        # 4. Labels for training
        y_real = np.ones((len(batch_index), 1))
        y_fake = np.zeros((len(batch_index), 1))
        y = np.concatenate((y_real, y_fake))

        # 5. Train Discriminator
        self.discriminator.train_on_batch([X], y)

        # Train ADVERSARIAL Network
        # 1. Prepare input - create a new batch of noise
        X_noise = np.random.uniform(-1, 1, [len(batch_index), C.noise_dimension]).astype('float32')
        X_fake_labels = np.eye(C.num_labels)[np.random.choice(C.num_labels, len(batch_index))]

        y = np.ones([len(batch_index), 1])

        self.adversarial.train_on_batch([X_noise, X_fake_labels], y)

    def get_generator_weights(self):
        return self.generator.get_weights()

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)

    def get_discriminator_weights(self):
        return self.discriminator.get_weights()

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)

    def get_sample_images(self):
        noise = np.random.uniform(-1, 1, [16, C.noise_dimension])
        fake_labels = np.eye(C.num_labels)[np.random.choice(C.num_labels, 16)]
        image_arrays = self.generator.predict([noise, fake_labels])

        return image_arrays


class DistributedGan(object):
    def __init__(self, distribution_size, real_data,real_labels,generator, discriminator):
        self.discriminator = discriminator
        self.generator = generator
        self.gans = [Gan(real_data[np.random.choice(len(real_data), len(real_data))],
                         real_labels[np.random.choice(len(real_labels), len(real_labels))],
                         k.models.clone_model(generator),
                         k.models.clone_model(discriminator),
                         C.learning_rate * distribution_size) for _ in range(distribution_size)]

    def get_generator_weights(self):
        weights_set = [gan.get_generator_weights() for gan in self.gans]
        return np.mean(weights_set, axis=0)

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)
        [gan.set_generator_weights(weights) for gan in self.gans]

    def get_discriminator_weights(self):
        weights_set = ([gan.get_discriminator_weights() for gan in self.gans])
        return np.mean(weights_set, axis=0)

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)
        [gan.set_discriminator_weights(weights) for gan in self.gans]

    def train_on_batch_index(self, batch_index):
        [gan.train_on_batch_index(batch_index) for gan in self.gans]
        self.set_generator_weights(self.get_generator_weights())
        self.set_discriminator_weights(self.get_discriminator_weights())

    def save_generator(self, directory_path, iteration_number):
        generator_path = directory_path + '/generator'
        os.makedirs(generator_path, exist_ok=True)

        self.generator.save_weights(generator_path + '/weights.h5')
        self.generator.save(generator_path+ '/model')
        with open(generator_path + '/architecture.json', 'w') as f:
            f.write(self.generator.to_json())
        k.utils.plot_model(self.generator, generator_path + '/graph.png', show_shapes=True)

        image_path = directory_path + '/image'

        image_arrays = self.gans[0].get_sample_images()

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
        self.discriminator.save(discriminator_path +'/model')

        with open(discriminator_path + '/architecture.json', 'w') as f:
            f.write(self.discriminator.to_json())
        k.utils.plot_model(self.discriminator, discriminator_path + '/graph.png', show_shapes=True)


class MultiDistributedGan(object):
    def __init__(self, real_data,real_labels, distribution_size, generators, discriminators):
        self.same_generator_gans_group = [[] for _ in range(len(generators))]
        self.same_discriminator_gans_group = [[] for _ in range(len(discriminators))]
        self.gans = []

        for i in range(len(generators)):
            for j in range(len(discriminators)):
                distributed_gan = DistributedGan(distribution_size,
                                                 real_data[np.random.choice(len(real_data), len(real_data))],
                                                 real_labels[np.random.choice(len(real_labels), len(real_labels))],
                                                 k.models.clone_model(generators[i]),
                                                 k.models.clone_model(discriminators[j]))

                self.gans.append(distributed_gan)
                self.same_generator_gans_group[i].append(distributed_gan)
                self.same_discriminator_gans_group[j].append(distributed_gan)

    def get_generators_weights(self):
        generators_weights = []
        for same_generator_gans in self.same_generator_gans_group:
            weights_set = [gan.get_generator_weights() for gan in same_generator_gans]
            generators_weights.append(np.mean(weights_set, axis=0))

        return generators_weights

    def set_generators_weights(self, generators_weights):
        for same_generator_gans, generator_weights in zip(self.same_generator_gans_group, generators_weights):
            [gan.set_generator_weights(generator_weights) for gan in same_generator_gans]

    def get_discriminators_weights(self):
        discriminators_weights = []
        for same_discriminator_gans in self.same_discriminator_gans_group:
            weights_set = []
            for same_discriminator_gan in same_discriminator_gans:
                weights_set.append(same_discriminator_gan.get_discriminator_weights())
            discriminators_weights.append(np.mean(np.array(weights_set), axis=0))

        return discriminators_weights

    def set_discriminators_weights(self, discriminators_weights):
        for same_discriminator_gans, discriminator_weights \
                in zip(self.same_discriminator_gans_group, discriminators_weights):
            [gan.set_discriminator_weights(discriminator_weights) for gan in same_discriminator_gans]

    def train_on_batch_index(self, batch_index):
        [gan.train_on_batch_index(batch_index) for gan in self.gans]
        self.set_generators_weights(self.get_generators_weights())
        self.set_discriminators_weights(self.get_discriminators_weights())

    def save(self, iteration_number):
        for i in range(len(self.same_generator_gans_group)):
            distributed_gan = self.same_generator_gans_group[i][0]
            distributed_gan.save_generator(C.path + '/generator %d' % i, iteration_number)

        for i in range(len(self.same_discriminator_gans_group)):
            distributed_gan = self.same_discriminator_gans_group[i][0]
            distributed_gan.save_discriminator(C.path + '/discriminator %d' % i)


def main():
    print("started training", flush=True)
    real_data = Data().real_data
    real_labels = Data().real_labels

    make_generator_function = C.make_fixed_generator
    make_discriminator_function = C.make_fixed_discriminator

    generator_size = 1

    discriminator_size = 1

    distribution_size = 1

    iteration_size = 100

    # to use loaded models use utils.load_discriminator(id) or utils.load_generator(id)
    generators = [make_generator_function(C.noise_inputs, C.label_inputs) for _ in range(generator_size)]
    discriminators = [make_discriminator_function(C.image_inputs) for _ in range(discriminator_size)]

    multi_distributed_gan = MultiDistributedGan(real_data,real_labels, distribution_size, generators, discriminators)

    for i in range(iteration_size):
        if i % 10 == 0:
            multi_distributed_gan.save(i)
            print('iteration', i)

        batch_indexes = np.array_split(np.random.permutation(len(real_data)), int(len(real_data) / C.batch_size))
        for batch_index in batch_indexes:
            multi_distributed_gan.train_on_batch_index(batch_index)

def test_generator(generator, label):
    test_noise_input = np.random.uniform(-1, 1 , size =[16, 100])
    test_fake_labels = np.eye(10)[np.repeat(label, 16)]
    fake_images = generator.predict([test_noise_input, test_fake_labels])
    plot_acc_images(fake_images)

if __name__ == '__main__':
    main()