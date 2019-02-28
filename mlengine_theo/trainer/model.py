from keras.layers import Concatenate, Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input, Subtract, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
import keras.optimizers as optimizers
import keras.backend as K
import tensorflow as tf
import pdb


print('warn default shapes  (duplicate) in model.py')
global_shape = (256, 256, 3)
local_shape = (128, 128, 3)

# hyperparameters
input_shape = (256, 256, 3)


def full_disc_layer(params, global_shape, local_shape, full_img, clip_coords):
    # the discriminator side
    disc_model = model_discriminator(global_shape, local_shape)

    disc_model = disc_model([full_img, clip_coords])
    # disc_model

    disc_brain = Model(inputs=[full_img, clip_coords], outputs=disc_model)
    disc_brain.compile(loss=params.disc_loss,
                        optimizer=getattr(optimizers, params.optimizer)(lr=params.learning_rate))
                        # optimizer=optimizers.Adadelta(lr=0.01))
    if params.verbosity == 'INFO':
        print(disc_model)
        disc_brain.summary()
        # view_models(disc_brain, '../summaries/disc_brain.png')

    return disc_brain, disc_model


def full_gen_layer(params, full_img, mask, erased_image):

    # view our net
    gen_model = model_generator(global_shape)

    # pass in the erased_image as input
    gen_model = gen_model(erased_image)

    gen_brain = Model(inputs=[full_img, mask, erased_image], outputs=gen_model)

    gen_brain.compile(
        loss=params.gen_loss,
        optimizer=getattr(optimizers, params.optimizer)(lr=params.learning_rate)
        # optimizer=optimizers.Adadelta(lr=0.001)
    )

    if params.verbosity == 'INFO':
        print(gen_model)
        print(gen_model)
        print(gen_brain)
        # view_models(gen_brain, '../summaries/gen_brain.png')
        gen_brain.summary()

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