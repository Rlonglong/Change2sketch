import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

import tensorflow_addons as tfa
import cv2, os

autotune = tf.data.AUTOTUNE



def get_model():
    cur_path = os.getcwd()
    # Size of the random crops to be used during training.
    input_img_size = (256, 256, 3)
    # Weights initializer for the layers.
    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # Gamma initializer for instance normalization.
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    class ReflectionPadding2D(layers.Layer):

        def __init__(self, padding=(1, 1), **kwargs):
            self.padding = tuple(padding)
            super().__init__(**kwargs)

        def call(self, input_tensor, mask=None):
            padding_width, padding_height = self.padding
            padding_tensor = [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ]
            return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def residual_block(
        x,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        gamma_initializer=gamma_init,
        use_bias=False,
    ):
        dim = x.shape[-1]
        input_tensor = x

        x = ReflectionPadding2D()(input_tensor)
        x = layers.Conv2D(
            dim,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = activation(x)

        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(
            dim,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.add([input_tensor, x])
        return x


    def downsample(
        x,
        filters,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
    ):
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        if activation:
            x = activation(x)
        return x


    def upsample(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_init,
        gamma_initializer=gamma_init,
        use_bias=False,
    ):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        if activation:
            x = activation(x)
        return x

    def get_resnet_generator(
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsample_blocks=2,
        gamma_initializer=gamma_init,
        name=None,
    ):
        img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
        x = ReflectionPadding2D(padding=(3, 3))(img_input)
        x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
            x
        )
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = downsample(x, filters=filters, activation=layers.Activation("relu"))

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = residual_block(x, activation=layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = upsample(x, filters, activation=layers.Activation("relu"))

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(3, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        model = keras.models.Model(img_input, x, name=name)
        return model
    
    gen_G = get_resnet_generator(name="generator_G")
    gen_G.load_weights(cur_path+"/_internal/model.weight", by_name=False)

    return gen_G


def get_predict(gen_G, path):
    # Size of the random crops to be used during training.
    input_img_size = (256, 256, 3)
    # Weights initializer for the layers.
    images = []
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))[:, :, :3]
    images.append(img)

    tensor = tf.convert_to_tensor(images, dtype=tf.uint8)
    labels_tensor = tf.convert_to_tensor([0 for i in range(len(images))], dtype=tf.int64)

    def normalize_img(img):
        img = tf.cast(img, dtype=tf.float32)
        # Map values in the range [-1, 1]
        return (img / 127.5) - 1.0
    def preprocess_image(img, label):
        img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
        img = normalize_img(img)
        return img

    images = tf.data.Dataset.from_tensor_slices((tensor, labels_tensor))

    batch_size = 1
    images = (
        images.map(preprocess_image, num_parallel_calls=autotune)
        .cache()
        .batch(batch_size)
    )

    for img in images:
        prediction = gen_G(img)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        return prediction


    