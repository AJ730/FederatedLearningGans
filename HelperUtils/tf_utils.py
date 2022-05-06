import os
import math
import numpy as np
from keras.models import load_model

from matplotlib import pyplot as plt



def plot_images(fake_images, generator_path, iteration_nmr):
    plt.figure(figsize=(5, 5))
    num_images = fake_images.shape[0]

    image_size = fake_images.shape[1]
    rows = int(math.sqrt(fake_images.shape[0]))

    os.makedirs(generator_path + "/batches", exist_ok=True)
    for j in range(num_images):
        plt.subplot(rows, rows + 1, j + 1)
        pltimg = np.reshape(fake_images[j], [image_size, image_size])
        plt.imshow(pltimg, cmap='gray')
        plt.axis('off')

    plt.savefig(generator_path + "/batches/batch" + str(iteration_nmr) + ".png")


def load_weights_predict(model, id, C):
    noise = np.random.uniform(-1, 1, [16, C.noise_dimension])
    model.load_weights(C.path + '/generator ' + str(id) + '/generator/weights.h5')
    image_array = model.predict(noise)
    plot_images(image_array, C.path + '/generator ' + str(id) + '/loaded', id)
    plt.show()

def load_model_predict(filename, id, C):
    noise = np.random.uniform(-1, 1, [16, C.noise_dimension])
    model= load_model(filename)
    image_array = model.predict(noise)
    plot_images(image_array, C.path + '/generator ' + str(id) + '/loaded', id)
    plt.show()

def load_discriminator(id, C):
    model = load_model(C.path + '/discriminator ' + str(id) + '/discriminator/discriminator')
    return model

def load_generator(id, C):
    model = load_model(C.path + '/generator ' + str(id) + '/generator/generator')
    return model