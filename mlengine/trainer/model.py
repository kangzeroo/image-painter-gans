import numpy as np
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input, Subtract, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.engine.network import Network
from keras.optimizers import Adadelta
import keras.backend as K
import tensorflow as tf
from tensorflow.python.lib.io import file_io



# Create the primitive generator net
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
    model_gen.name = 'Primitive-Generator'
    return model_gen

# Create the primitive discriminator net
def model_discriminator(GLOBAL_SHAPE=(256, 256, 3), LOCAL_SHAPE=(128, 128, 3)):
    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                             crop[1],
                                             crop[0],
                                             crop[3] - crop[1],
                                             crop[2] - crop[0])

    in_pts = Input(shape=(4,), dtype='int32')
    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=LOCAL_SHAPE)
    g_img = Input(shape=GLOBAL_SHAPE)
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
    model_disc.name = 'Primitive-Discriminator'
    return model_disc

# upgrade the primitive net to an augmented "full_gen_layer" net
# simply has the inputs added
def full_gen_layer(FULL_IMG, MASK, ONES, GLOBAL_SHAPE, OPTIMIZER):
    from keras.layers import Concatenate

    # grab the INVERSE_MASK, that only shows the MASKed areas
    # 1 - MASK
    INVERSE_MASK = Subtract()([ONES, MASK])

    # which outputs the erased_image as input
    # FULL_IMG * (1 - MASK)
    erased_image = Multiply()([FULL_IMG, INVERSE_MASK])

    # view our net
    gen_model = model_generator(GLOBAL_SHAPE)
    # print(gen_model)

    # pass in the erased_image as input
    gen_model = gen_model(erased_image)
    # print(gen_model)

    gen_brain = Model(inputs=[FULL_IMG, MASK, ONES], outputs=gen_model)

    gen_brain.compile(
        loss='mse',
        optimizer=OPTIMIZER
    )
    return gen_brain, gen_model

# connect the primitive discriminator net to the output of the augmented net
def full_disc_layer(GLOBAL_SHAPE, LOCAL_SHAPE, FULL_IMG, CLIP_COORDS, OPTIMIZER):
    # the discriminator side
    disc_model = model_discriminator(GLOBAL_SHAPE, LOCAL_SHAPE)

    disc_model = disc_model([FULL_IMG, CLIP_COORDS])
    disc_model
    # print(disc_model)

    disc_brain = Model(inputs=[FULL_IMG, CLIP_COORDS], outputs=disc_model)
    disc_brain.compile(loss='binary_crossentropy',
                        optimizer=OPTIMIZER)
    return disc_brain, disc_model

# get the gen brain and disc brain
def getBrains(
        GLOBAL_SHAPE,
        LOCAL_SHAPE,
        OPTIMIZER,

        FULL_IMG,
        MASK,
        ONES,
        CLIP_COORDS,

        ALPHA,

        BUCKET_NAME,
        JOB_DIR,
    ):
    gen_brain, gen_model = full_gen_layer(FULL_IMG, MASK, ONES, GLOBAL_SHAPE, OPTIMIZER)
    disc_brain, disc_model = full_disc_layer(GLOBAL_SHAPE, LOCAL_SHAPE, FULL_IMG, CLIP_COORDS, OPTIMIZER)

    # the final brain
    disc_model.trainable = False
    connected_disc = Model(inputs=[FULL_IMG, CLIP_COORDS], outputs=disc_model)
    connected_disc.name = 'Connected-GANs'
    # print(connected_disc)

    brain = Model(inputs=[FULL_IMG, MASK, ONES, CLIP_COORDS], outputs=[gen_model, connected_disc([gen_model, CLIP_COORDS])])
    brain.compile(loss=['mse', 'binary_crossentropy'],
                          loss_weights=[1.0, ALPHA], optimizer=OPTIMIZER)
    print(brain.summary())
    # with file_io.FileIO(f'gs://{BUCKET_NAME}/{JOB_DIR}brain_summary/brain_summary.txt', 'wb') as f:
    #     f.write(brain.summary())
    return brain, gen_brain, disc_brain
