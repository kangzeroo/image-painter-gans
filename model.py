import numpy as np
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input, Subtract, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.engine.network import Network
from keras.optimizers import Adadelta
import keras.backend as K
import tensorflow as tf


def view_models(model, filename):
    from keras.utils import plot_model
    plot_model(model, to_file=filename, show_shapes=True)


def full_disc_layer(global_shape, local_shape, full_img, clip_coords):
    # the discriminator side
    disc_model = model_discriminator(global_shape, local_shape)

    disc_model = disc_model([full_img, clip_coords])
    disc_model
    # print(disc_model)

    disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)
    disc_brain.compile(loss='binary_crossentropy',
                        optimizer=optimizer)
    # disc_brain.summary()
    view_models(disc_brain, 'summaries/disc_brain.png')
    return disc_brain, disc_model


def full_gen_layer(full_img, mask, ones):
    from keras.layers import Concatenate

    # grab the inverse mask, that only shows the masked areas
    # 1 - mask
    inverse_mask = Subtract()([ones, mask])

    # which outputs the erased_image as input
    # full_img * (1 - mask)
    erased_image = Multiply()([full_img, inverse_mask])

    # view our net
    gen_model = model_generator(global_shape)
    # print(gen_model)

    # pass in the erased_image as input
    gen_model = gen_model(erased_image)
    # print(gen_model)

    gen_brain = Model(inputs=[full_img, mask, ones], outputs=gen_model)
    # print(gen_brain)
    view_models(gen_brain, 'summaries/gen_brain.png')

    gen_brain.compile(
        loss='mse',
        optimizer=optimizer
    )
    # gen_brain.summary()
    return gen_brain, gen_model


def model_generator(input_shape=(256, 256, 3)):
    in_layer = Input(shape=input_shape)

    model = Conv2D(64, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1))(in_layer)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(128, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(4, 4))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(8, 8))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(16, 16))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(128, kernel_size=4, strides=2,
                              padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(64, kernel_size=4, strides=2,
                              padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    model = Conv2D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1))(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    model_gen = Model(inputs=in_layer, outputs=model)
    model_gen.name = 'Gener8tor'
    return model_gen


def model_discriminator(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):
    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                             crop[1],
                                             crop[0],
                                             crop[3] - crop[1],
                                             crop[2] - crop[0])

    in_pts = Input(shape=(4,), dtype='int32')
    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=local_shape)
    g_img = Input(shape=global_shape)
    l_img = cropping([g_img, in_pts])

    # Local Discriminator
    x_l = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Flatten()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    x_g = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(1024, activation='relu')(x_g)

    x = Concatenate(axis=1)([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    model_disc = Model(inputs=[g_img, in_pts], outputs=x)
    model_disc.name = 'Discimi-hater'
    return model_disc