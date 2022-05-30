from math import floor

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from numpy import asarray
from numpy import exp
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from skimage.transform import resize


# scale an array of images to a new size
from helperUtils.tf_utils import load_generator_result
from tensor_flow.DistributedGanCPUTensorflow import C


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load custom trained model
    model = tf.keras.models.load_model('./classifier/MNIST/model')
    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # predict p(y|x)
        p_yx = model.predict(subset)

        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


if __name__ == '__main__':

    model = load_generator_result(0, '../results_federated_learning_tf', 'MNIST', offline=True, architecure='FLGAN',
                                  clients=2)

    print("loaded model")
    noise = np.random.uniform(-1, 1, [10000, C.noise_dimension])
    imgs = model.predict(noise)

    # plt.imshow(imgs[0], cmap='gray')
    # plt.show()
    #
    #
    # model2 = tf.keras.models.load_model('./classifier/MNIST/model')
    # model2 = tf.keras.Sequential([model2, tf.keras.layers.Softmax()])
    # predictions = model2.predict(imgs)
    # print(np.argmax(predictions[0]))

    is_avg, is_std = calculate_inception_score(imgs)

    print('score', is_avg, is_std)

    # model = tf.keras.models.load_model('./classifier/MNIST/model')



    # (real_data, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # # Make it 3D dataset
    # real_data = real_data[:5000].astype('float32')
    # real_data = np.reshape(real_data, [-1, C.image_width, C.image_width, 1])
    #     # Standardize data : 0 to 1
    # real_data = real_data.astype('float32') / 255
    #
    #
    # print(np.argmax(predictions[0]))
    # plt.imshow(real_data[0], cmap='gray')
    # plt.show()

